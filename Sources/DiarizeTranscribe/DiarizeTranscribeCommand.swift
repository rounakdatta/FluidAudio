import AVFoundation
import FluidAudio
import Foundation

/// Result structure for combined diarization + transcription
struct DiarizedTranscript: Codable {
    let segments: [SpeakerSegment]
    let metadata: TranscriptMetadata

    struct SpeakerSegment: Codable {
        let speaker: String
        let text: String
        let startTime: Float
        let endTime: Float
        let words: [WordTiming]?
        let confidence: Float
    }

    struct WordTiming: Codable {
        let word: String
        let startTime: Float
        let endTime: Float
        let confidence: Float
    }

    struct TranscriptMetadata: Codable {
        let audioFile: String
        let durationSeconds: Float
        let speakerCount: Int
        let speakers: [String]
        let processingTime: TimeInterval
        let diarizationTime: TimeInterval
        let transcriptionTime: TimeInterval
        let clusteringThreshold: Float
        let modelVersion: String
    }
}

/// Command to perform combined diarization and transcription
@available(macOS 13.0, *)
enum DiarizeTranscribeCommand {
    private static let logger = AppLogger(category: "DiarizeTranscribe")

    static func run(arguments: [String]) async {
        guard !arguments.isEmpty else {
            logger.error("No audio file specified")
            printUsage()
            exit(1)
        }

        let audioFile = arguments[0]
        var threshold: Float = 0.7
        var outputFile: String?
        var modelVersion: AsrModelVersion = .v3
        var includeWordTimings = true

        // Parse options
        var i = 1
        while i < arguments.count {
            switch arguments[i] {
            case "--threshold":
                if i + 1 < arguments.count {
                    threshold = Float(arguments[i + 1]) ?? 0.7
                    i += 1
                }
            case "--output":
                if i + 1 < arguments.count {
                    outputFile = arguments[i + 1]
                    i += 1
                }
            case "--model-version":
                if i + 1 < arguments.count {
                    switch arguments[i + 1].lowercased() {
                    case "v2", "2":
                        modelVersion = .v2
                    case "v3", "3":
                        modelVersion = .v3
                    default:
                        logger.error("Invalid model version: \(arguments[i + 1]). Use 'v2' or 'v3'")
                        exit(1)
                    }
                    i += 1
                }
            case "--no-word-timings":
                includeWordTimings = false
            case "--help", "-h":
                printUsage()
                exit(0)
            default:
                logger.warning("Unknown option: \(arguments[i])")
            }
            i += 1
        }

        logger.info("🎙️  Processing audio file: \(audioFile)")
        logger.info("   Clustering threshold: \(threshold)")
        logger.info("   Model version: \(modelVersion == .v2 ? "v2 (English)" : "v3 (Multilingual)")")

        do {
            let result = try await processAudioFile(
                path: audioFile,
                threshold: threshold,
                modelVersion: modelVersion,
                includeWordTimings: includeWordTimings
            )

            // Output results
            if let outputFile = outputFile {
                try saveResults(result, to: outputFile)
                logger.info("💾 Results saved to: \(outputFile)")
            } else {
                printResults(result)
            }

        } catch {
            logger.error("Failed to process audio: \(error)")
            exit(1)
        }
    }

    private static func processAudioFile(
        path: String,
        threshold: Float,
        modelVersion: AsrModelVersion,
        includeWordTimings: Bool
    ) async throws -> DiarizedTranscript {

        let totalStartTime = Date()

        // Load audio once (shared for both diarization and ASR)
        logger.info("📂 Loading audio file...")
        let audioSamples = try AudioConverter().resampleAudioFile(path: path)
        let duration = Float(audioSamples.count) / 16000.0
        logger.info("   Loaded \(audioSamples.count) samples (\(String(format: "%.2f", duration))s)")

        // Step 1: Run diarization
        logger.info("\n🎯 Step 1/3: Running speaker diarization...")
        let diarizationStartTime = Date()

        let diarizationConfig = DiarizerConfig(
            clusteringThreshold: threshold,
            debugMode: false
        )
        let diarizerManager = DiarizerManager(config: diarizationConfig)
        let diarizationModels = try await DiarizerModels.downloadIfNeeded()
        diarizerManager.initialize(models: diarizationModels)

        let diarizationResult = try diarizerManager.performCompleteDiarization(
            audioSamples,
            sampleRate: 16000
        )

        let diarizationTime = Date().timeIntervalSince(diarizationStartTime)
        logger.info(
            "   ✓ Found \(diarizationResult.segments.count) segments from \(Set(diarizationResult.segments.map(\.speakerId)).count) speakers"
        )
        logger.info("   ✓ Processing time: \(String(format: "%.2f", diarizationTime))s")

        // Step 2: Run transcription
        logger.info("\n📝 Step 2/3: Running speech recognition...")
        let transcriptionStartTime = Date()

        let asrModels = try await AsrModels.downloadAndLoad(version: modelVersion)
        let asrManager = AsrManager(config: .default)
        try await asrManager.initialize(models: asrModels)

        let transcriptionResult = try await asrManager.transcribe(audioSamples, source: .system)

        let transcriptionTime = Date().timeIntervalSince(transcriptionStartTime)
        logger.info("   ✓ Transcription complete: \(transcriptionResult.text.prefix(100))...")
        logger.info("   ✓ Processing time: \(String(format: "%.2f", transcriptionTime))s")

        // Step 3: Merge diarization + transcription
        logger.info("\n🔗 Step 3/3: Merging speaker labels with transcript...")
        let mergedSegments = mergeSpeakerAndTranscript(
            diarizationSegments: diarizationResult.segments,
            transcriptionResult: transcriptionResult,
            includeWordTimings: includeWordTimings
        )

        let totalTime = Date().timeIntervalSince(totalStartTime)

        logger.info("   ✓ Created \(mergedSegments.count) speaker-attributed segments")
        logger.info("\n⏱️  Total processing time: \(String(format: "%.2f", totalTime))s")
        logger.info("   RTFx: \(String(format: "%.2f", Double(duration) / totalTime))x")

        // Build metadata
        let speakers = Array(Set(diarizationResult.segments.map(\.speakerId))).sorted()
        let metadata = DiarizedTranscript.TranscriptMetadata(
            audioFile: path,
            durationSeconds: duration,
            speakerCount: speakers.count,
            speakers: speakers,
            processingTime: totalTime,
            diarizationTime: diarizationTime,
            transcriptionTime: transcriptionTime,
            clusteringThreshold: threshold,
            modelVersion: modelVersion == .v2 ? "v2" : "v3"
        )

        return DiarizedTranscript(segments: mergedSegments, metadata: metadata)
    }

    private static func mergeSpeakerAndTranscript(
        diarizationSegments: [TimedSpeakerSegment],
        transcriptionResult: ASRResult,
        includeWordTimings: Bool
    ) -> [DiarizedTranscript.SpeakerSegment] {

        guard let tokenTimings = transcriptionResult.tokenTimings, !tokenTimings.isEmpty else {
            // Fallback: No word timings available, create one segment per speaker
            return diarizationSegments.map { segment in
                DiarizedTranscript.SpeakerSegment(
                    speaker: segment.speakerId,
                    text: "",  // No text mapping without timings
                    startTime: segment.startTimeSeconds,
                    endTime: segment.endTimeSeconds,
                    words: nil,
                    confidence: transcriptionResult.confidence
                )
            }
        }

        // Build a map of speaker segments for fast lookup
        var currentSegmentIndex = 0
        var mergedSegments: [DiarizedTranscript.SpeakerSegment] = []
        var currentSpeaker: String?
        var currentText = ""
        var currentWords: [DiarizedTranscript.WordTiming] = []
        var currentStartTime: Float?
        var currentEndTime: Float = 0

        for tokenTiming in tokenTimings {
            // Find which speaker is active at this token's timestamp
            let tokenMidpoint = Float((tokenTiming.startTime + tokenTiming.endTime) / 2)

            // Find the speaker segment that contains this token
            while currentSegmentIndex < diarizationSegments.count {
                let segment = diarizationSegments[currentSegmentIndex]

                if tokenMidpoint >= segment.startTimeSeconds && tokenMidpoint <= segment.endTimeSeconds {
                    // Token belongs to this speaker
                    let speakerId = segment.speakerId

                    if currentSpeaker != speakerId {
                        // Speaker changed - save previous segment
                        if let speaker = currentSpeaker, !currentText.isEmpty {
                            mergedSegments.append(
                                DiarizedTranscript.SpeakerSegment(
                                    speaker: speaker,
                                    text: currentText.trimmingCharacters(in: .whitespaces),
                                    startTime: currentStartTime ?? 0,
                                    endTime: currentEndTime,
                                    words: includeWordTimings ? currentWords : nil,
                                    confidence: transcriptionResult.confidence
                                )
                            )
                        }

                        // Start new segment
                        currentSpeaker = speakerId
                        currentText = tokenTiming.token
                        currentWords =
                            includeWordTimings
                            ? [
                                DiarizedTranscript.WordTiming(
                                    word: tokenTiming.token,
                                    startTime: Float(tokenTiming.startTime),
                                    endTime: Float(tokenTiming.endTime),
                                    confidence: tokenTiming.confidence
                                )
                            ] : []
                        currentStartTime = Float(tokenTiming.startTime)
                        currentEndTime = Float(tokenTiming.endTime)
                    } else {
                        // Same speaker - append token
                        currentText += tokenTiming.token
                        if includeWordTimings {
                            currentWords.append(
                                DiarizedTranscript.WordTiming(
                                    word: tokenTiming.token,
                                    startTime: Float(tokenTiming.startTime),
                                    endTime: Float(tokenTiming.endTime),
                                    confidence: tokenTiming.confidence
                                )
                            )
                        }
                        currentEndTime = Float(tokenTiming.endTime)
                    }
                    break
                } else if tokenMidpoint < segment.startTimeSeconds {
                    // Token is before this segment (silence/gap)
                    break
                } else {
                    // Token is after this segment - move to next
                    currentSegmentIndex += 1
                }
            }
        }

        // Add final segment
        if let speaker = currentSpeaker, !currentText.isEmpty {
            mergedSegments.append(
                DiarizedTranscript.SpeakerSegment(
                    speaker: speaker,
                    text: currentText.trimmingCharacters(in: .whitespaces),
                    startTime: currentStartTime ?? 0,
                    endTime: currentEndTime,
                    words: includeWordTimings ? currentWords : nil,
                    confidence: transcriptionResult.confidence
                )
            )
        }

        return mergedSegments
    }

    private static func saveResults(_ result: DiarizedTranscript, to path: String) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(result)

        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        try data.write(to: url)
    }

    private static func printResults(_ result: DiarizedTranscript) {
        logger.info("\n" + String(repeating: "=", count: 80))
        logger.info("SPEAKER-DIARIZED TRANSCRIPT")
        logger.info(String(repeating: "=", count: 80))

        for (index, segment) in result.segments.enumerated() {
            let timeRange = String(format: "[%.2f - %.2f]", segment.startTime, segment.endTime)
            logger.info("\n[\(index + 1)] \(segment.speaker) \(timeRange)")
            logger.info("    \(segment.text)")

            if let words = segment.words, !words.isEmpty {
                logger.info("    Words: \(words.count)")
            }
        }

        logger.info("\n" + String(repeating: "=", count: 80))
        logger.info("METADATA")
        logger.info(String(repeating: "=", count: 80))
        logger.info("Duration: \(String(format: "%.2f", result.metadata.durationSeconds))s")
        logger.info("Speakers: \(result.metadata.speakers.joined(separator: ", "))")
        logger.info("Total segments: \(result.segments.count)")
        logger.info("Processing time: \(String(format: "%.2f", result.metadata.processingTime))s")
        logger.info(
            "RTFx: \(String(format: "%.2f", Double(result.metadata.durationSeconds) / result.metadata.processingTime))x"
        )
    }

    private static func printUsage() {
        logger.info(
            """

            Diarize-Transcribe Command Usage:
                fluidaudio diarize-transcribe <audio_file> [options]

            Description:
                Performs speaker diarization and speech recognition in one pass,
                producing a speaker-attributed transcript with timestamps.

            Options:
                --threshold <float>           Clustering threshold for speaker separation (default: 0.7)
                --output <file>              Save results to JSON file (default: print to stdout)
                --model-version <version>    ASR model: v2 (English) or v3 (Multilingual) (default: v3)
                --no-word-timings            Exclude word-level timings from output
                --help, -h                   Show this help message

            Examples:
                # Basic usage
                fluidaudio diarize-transcribe podcast.mp3

                # Save to JSON file
                fluidaudio diarize-transcribe podcast.mp3 --output transcript.json

                # Use English-only model with custom threshold
                fluidaudio diarize-transcribe podcast.mp3 --model-version v2 --threshold 0.8

            Output Format:
                {
                  "segments": [
                    {
                      "speaker": "Speaker_01",
                      "text": "Hello everyone, welcome to the show.",
                      "startTime": 0.0,
                      "endTime": 5.2,
                      "words": [...],
                      "confidence": 0.95
                    }
                  ],
                  "metadata": {
                    "durationSeconds": 3600.0,
                    "speakerCount": 2,
                    "speakers": ["Speaker_01", "Speaker_02"],
                    ...
                  }
                }
            """
        )
    }
}
