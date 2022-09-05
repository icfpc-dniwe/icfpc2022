{-# LANGUAGE StrictData #-}

module ICFPC2022.Solver.Pixelize
  ( solve
  ) where

import Linear
import Control.Monad.Writer.Strict
import qualified Data.Massiv.Array as A

import ICFPC2022.Types
import ICFPC2022.Commands

type Solver a = Writer Program a

solveOne :: Picture -> Point -> Size -> BlockId -> Solver ()
solveOne picture offset@(V2 offsetX offsetY) size currentBlock = do
  let px = picture `A.evaluate'` pointToIx offset
      pictureSlice = A.extract' (pointToIx offset) (pointToSize size) picture

  case size of
    V2 0 _ -> error "Zero-size image"
    V2 _ 0 -> error "Zero-size image"
    _ | A.all (== px) pictureSlice ->
        tell [ Color { colorBlock = currentBlock
                     , colorValue = px
                     }
             ]
    V2 width 1 -> do
      let midW = width `div` 2
      tell [ LineCut { lineCutBlock = currentBlock
                     , lineCutOrientation = CutX
                     , lineCutOffset = offsetX + midW
                     }
           ]
      solveOne picture offset (V2 midW 1) (currentBlock ++ [0])
      solveOne picture (offset + V2 midW 0) (V2 (width - midW) 1) (currentBlock ++ [1])
    V2 1 height -> do
      let midH = height `div` 2
      tell [ LineCut { lineCutBlock = currentBlock
                     , lineCutOrientation = CutY
                     , lineCutOffset = offsetY + midH
                     }
           ]
      solveOne picture offset (V2 1 midH) (currentBlock ++ [0])
      solveOne picture (offset + V2 0 midH) (V2 1 (height - midH)) (currentBlock ++ [1])
    V2 width height -> do
      let midPoint@(V2 midW midH) = fmap (`div` 2) size
      tell [ PointCut { pointCutBlock = currentBlock
                      , pointCutPoint = offset + midPoint
                      }
           ]
      solveOne picture (offset + V2 0 0) midPoint (currentBlock ++ [0])
      solveOne picture (offset + V2 midW 0) (V2 (width - midW) midH) (currentBlock ++ [1])
      solveOne picture (offset + midPoint) (V2 (width - midW) (height - midH)) (currentBlock ++ [2])
      solveOne picture (offset + V2 0 midH) (V2 midW (height - midH)) (currentBlock ++ [3])

solve :: Picture -> Program
solve picture = execWriter $ solveOne picture 0 (sizeToPoint $ A.size picture) [0]
