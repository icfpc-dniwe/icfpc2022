module ICFPC2022.Picture where

import Linear
import Codec.Picture
import Data.Massiv.Array

import ICFPC2022.Types

convertImage :: DynamicImage -> Picture
convertImage (convertRGBA8 -> image@(Image {..})) =
  makeArray Seq (Sz2 imageWidth imageHeight) (\(x :. y) -> convertPixel $ pixelAt image x (imageHeight - y - 1))
  where convertPixel (PixelRGBA8 r g b a) = V4 r g b a
