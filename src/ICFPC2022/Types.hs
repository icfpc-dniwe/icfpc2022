{-# LANGUAGE StrictData #-}

module ICFPC2022.Types where

import Data.Word
import Linear
import Data.Massiv.Array

type I2 = V2 Int
type Point = I2

type Color = V4 Word8

type Picture = Array U Ix2 Color
