module ICFPC2022.Utils where

import ICFP2022.Types
import Linear.Metric

pixelDistance :: Color -> Color -> Float
pixelDistance left right = norm $ left - right

imageDistance = 0
