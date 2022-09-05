{-# LANGUAGE StrictData #-}

module ICFPC2022.Types where

import Data.Word
import Linear
import Data.Massiv.Array

type I2 = V2 Int
type Point = I2
type Size = I2

type Color = V4 Word8

type GPicture r = Array r Ix2 Color
type Picture = Array U Ix2 Color

pointToIx :: I2 -> Ix2
pointToIx (V2 x y) = Ix2 x y

ixToPoint :: Ix2 -> I2
ixToPoint (Ix2 x y) = V2 x y

pointToSize :: I2 -> Sz2
pointToSize (V2 x y) = Sz2 x y

sizeToPoint :: Sz2 -> I2
sizeToPoint (Sz2 x y) = V2 x y
