{-# LANGUAGE StrictData #-}

module ICFPC2022.Field where

import ICFPC2022.Types

type Offset = Point

data BlockOrientation = BlockX | BlockY
                      deriving (Show, Eq)

data BlockTree = SimpleBlock { blockColor :: Color }
               | ComplexBlock { blockOrientation :: BlockOrientation
                              , blockOffset :: Int
                              , blockSubA :: BlockTree
                              , blockSubB :: BlockTree
                              }
               deriving (Show, Eq)

data Field = Field { fieldSize :: I2
                   , fieldBlocks :: BlockTree
                   }
           deriving (Show, Eq)
