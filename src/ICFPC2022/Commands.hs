{-# LANGUAGE StrictData #-}

module ICFPC2022.Commands where

import ICFPC2022.Types

type BlockId = Int

data CutOrientation = CutX | CutY
                    deriving (Show, Eq)

data LineCutMove = LineCutMove { lineCutBlock :: BlockId
                               , lineCutOrientation :: CutOrientation
                               , lineCutOffset :: Int
                               }
                 deriving (Show, Eq)

data PointCutMove = PointCutMove { pointCutBlock :: BlockId
                                 , pointCutPoint :: Point
                                 }
                  deriving (Show, Eq)

data ColorMove = ColorMove { colorBlock :: BlockId
                           , colorValue :: Color
                           }
               deriving (Show, Eq)

data SwapMove = SwapMove { swapBlockA :: BlockId
                         , swapBlockB :: BlockId
                         }
              deriving (Show, Eq)

data MergeMove = MergeMove { mergeBlockA :: BlockId
                           , mergeBlockB :: BlockId
                           }
               deriving (Show, Eq)

data Move = LineCut LineCutMove
          | PointCut PointCutMove
          | Color ColorMove
          | Swap SwapMove
          | Merge MergeMove
          deriving (Show, Eq)

type Program = [Move]

move_cost :: Move -> Int
move_cost (LineCut _) = 7
move_cost (PointCut _) = 10
move_cost (Color _) = 5
move_cost (Swap _) = 3
move_cost (Merge _) = 1
