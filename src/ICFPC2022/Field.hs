{-# LANGUAGE StrictData #-}

module ICFPC2022.Field where

import ICFPC2022.Types
import Data.Set (Set)

type Offset = Point

data SimpleBlock = SimpleBlock {
      simple_offset :: Offset
    , color :: Color
} deriving (Show, Eq)

type ChildBlocks = Set SimpleBlock

data ComplexBlock = ComplexBlock {
      complex_offset :: Offset
    , childs :: ChildBlocks
} deriving (Show, Eq)

data Block = Simple SimpleBlock
           | Complex ComplexBlock
           deriving (Show, Eq)
