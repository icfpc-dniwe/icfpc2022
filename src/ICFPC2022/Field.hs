{-# LANGUAGE StrictData #-}

module ICFPC2022.Field where

import ICFPC2022.Types

data Block = SimpleBlock { blockColor :: Color }
           deriving (Show, Eq)
