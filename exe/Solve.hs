import System.Environment
import System.IO
import System.Exit
import Codec.Picture
import qualified Data.ByteString.Builder as Builder

import ICFPC2022.Picture
import ICFPC2022.PrettyPrint
import ICFPC2022.Solver.Dumb

main :: IO ()
main = do
  filePath <- getArgs >>= \case
    [p] -> return p
    _ -> do
      hPutStrLn stderr "Usage: solve path/to/image.png"
      exitFailure
  img <- readImage filePath >>= \case
    Right img -> return img
    Left e -> do
      hPutStrLn stderr $ "Failed to read image: " ++ e
      exitFailure

  let picture = convertImage img
  let program = solve picture
  putStrLn "# solution"
  Builder.hPutBuilder stdout $ printProgram program
