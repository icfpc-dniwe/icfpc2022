module ICFPC2022.Utils where

import ICFPC2022.Types
import Linear.Metric

pixelDistance :: Color -> Color -> Float
pixelDistance left right = norm $ fmap fromIntegral $ left - right

imageDistance = 0
