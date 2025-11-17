import FluidAudio
import Foundation

// Pass all arguments except the program name
let arguments = Array(CommandLine.arguments.dropFirst())
await DiarizeTranscribeCommand.run(arguments: arguments)
