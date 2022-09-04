module ICFPC2022.PrettyPrint where

import Data.List
import Data.ByteString.Builder

import Linear

import ICFPC2022.Types
import ICFPC2022.Commands

printOrientation :: CutOrientation -> Builder
printOrientation CutX = "[x]"
printOrientation CutY = "[y]"

printPoint :: Point -> Builder
printPoint (V2 x y) = "[" <> intDec x <> "," <> intDec y <> "]"

printColor :: Color -> Builder
printColor (V4 r g b a) =
  "[" <> word8Dec r <> "," <>
  word8Dec g <> "," <>
  word8Dec b <> "," <>
  word8Dec a <> "]"

printBlockId :: BlockId -> Builder
printBlockId blockId = "[" <> mconcat (intersperse "." $ map intDec blockId) <> "]"

printMove :: Move -> Builder
printMove (LineCut  {..}) =
  "cut " <> printBlockId lineCutBlock <> " " <>
  printOrientation lineCutOrientation <> " [" <>
  intDec lineCutOffset <> "]"
printMove (PointCut {..}) =
  "cut " <> printBlockId pointCutBlock <> " " <>
  printPoint pointCutPoint
printMove (Color    {..}) =
  "color " <> printBlockId colorBlock <> " " <> printColor colorValue
printMove (Swap     {..}) =
  "swap " <> printBlockId swapBlockA <> " " <> printBlockId swapBlockB
printMove (Merge    {..}) =
  "merge " <> printBlockId mergeBlockA <> " " <> printBlockId mergeBlockB

printProgram :: Program -> Builder
printProgram = mconcat . map (\move -> printMove move <> "\n")
