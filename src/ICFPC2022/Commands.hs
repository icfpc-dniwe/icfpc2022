{-# LANGUAGE StrictData #-}

module ICFPC2022.Commands where

import ICFPC2022.Types

type BlockNumber = Int
type BlockId = [BlockNumber]

data CutOrientation = CutX | CutY
                    deriving (Show, Eq)

data Move = LineCut { lineCutBlock :: BlockId
                    , lineCutOrientation :: CutOrientation
                    , lineCutOffset :: Int
                    }
          | PointCut { pointCutBlock :: BlockId
                     , pointCutPoint :: Point
                     }
          | Color { colorBlock :: BlockId
                  , colorValue :: Color
                  }
          | Swap { swapBlockA :: BlockId
                 , swapBlockB :: BlockId
                 }
          | Merge { mergeBlockA :: BlockId
                  , mergeBlockB :: BlockId
                  }
          deriving (Show, Eq)

type Program = [Move]

moveCost :: Move -> Int
moveCost (LineCut {}) = 7
moveCost (PointCut {}) = 10
moveCost (Color {}) = 5
moveCost (Swap {}) = 3
moveCost (Merge {}) = 1
