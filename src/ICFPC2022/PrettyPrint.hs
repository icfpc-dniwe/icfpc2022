module ICFPC2022.PrettyPrint where

import Data.ByteString.Builder

import Linear

import ICFPC2022.Types
import ICFPC2022.Commands

printOrientation :: CutOrientation -> Builder
printOrientation CutX = "x"
printOrientation CutY = "y"

printPoint :: Point -> Builder
printPoint (V2 x y) = "[" <> intDec x <> "," <> intDec y <> "]"

printColor :: Color -> Builder
printColor (V4 r g b a) =
  "[" <> word8Dec r <> "," <>
  word8Dec g <> "," <>
  word8Dec b <> "," <>
  word8Dec a <> "]"

printMove :: Move -> Builder
printMove (LineCut  (LineCutMove  {..})) =
  "cut " <> intDec lineCutBlock <> " " <>
  printOrientation lineCutOrientation <> " " <>
  intDec lineCutOffset
printMove (PointCut (PointCutMove {..})) =
  "cut " <> intDec pointCutBlock <> " " <>
  printPoint pointCutPoint
printMove (Color    (ColorMove    {..})) =
  "color " <> intDec colorBlock <> " " <> printColor colorValue
printMove (Swap     (SwapMove     {..})) =
  "swap " <> intDec swapBlockA <> " " <> intDec swapBlockB
printMove (Merge    (MergeMove    {..})) =
  "merge " <> intDec mergeBlockA <> " " <> intDec mergeBlockB

printProgram :: Program -> Builder
printProgram = mconcat . map (\move -> printMove move <> "\n")
