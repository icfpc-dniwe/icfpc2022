import System.Environment
import System.IO
import System.Exit
import Codec.Picture

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

  putStrLn "# empty program"
