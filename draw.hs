module Main where

import Graphics.HsCharts
import Graphics.Gloss
import Text.Parsec.Prim
import Text.Parsec.Combinator
import Text.Parsec.Char
import Text.Parsec.Token
import Debug.Trace
ps str = case (parse tpps "" (str :: String)) of
  Left err  -> trace (show err) []
  Right xs  -> xs

tpps = lne `endBy` (char '\n')
lne = (fmap (read :: String->Float) (many (digit <|> char '.'))) `sepBy` (char ',' )

 
-- if you parse a stream of characters
ps2 fn = do
  psed <- (fmap ps $ readFile fn)
  return $ fmap (\xs -> (xs!!0,xs!!2)) psed
main :: IO ()
main = do
    pairsv <-ps2 "stats/valid.txt"
    pairst <-ps2 "stats/train.txt"
    let disp = InWindow "HsCharts Demo" (windowW, windowH) (0, 0)
    display disp white $ pictures [
      q 0 0 $ mychart2 pairsv,
      q 1 0 $ mychart2 pairst]
    
windowW = 1800
windowH = 1000
chartW  = fromIntegral $  windowW 
chartH  = fromIntegral $ windowH 
chartM  = 50

q x y  = translate dx dy
    where dx = (-(fromIntegral windowW / 2)) + (chartM + x * (chartW + chartM))
          dy = (fromIntegral windowH / 2) - chartH - chartM - (y * (chartH + chartM))

bgColor      = makeColor 0.98 0.98 0.98 1
gridColor    = makeColor 0.8 0.8 0.8 1

gridAltColor = makeColor 0.8 0.8 0.8 0.4
pointColor   = makeColor 0.15 0.50 0.75 1
pointColor'  = makeColor 0.75 0.15 0.15 1
pointColor'' = makeColor 0.15 0.75 0.15 1
barColor     = makeColor 0.15 0.50 0.75 0.8
areaColor    = makeColor 0.15 0.50 0.75 0.4

-----------------------------------------------------------------------------
-- logScaleChart :: Picture
readlogs = do
  txt<-readFile "stats/valid.txt"
--  print txt
  let psed = ps txt
  print psed
  
mychart pairs =
    pictures [ color bgColor $ plotChartBackground xAxis yAxis
             , plotAxes xAxis yAxis
             , color pointColor $ plotAxisScales xAxis yAxis (1, 10)
             , color pointColor $ plotLineChart xAxis yAxis pairs
             ]
    where (xs,ys) = unzip pairs
          xAxis     = fixedScaleAxis Linear chartW 0 $ maximum xs
          yAxis     = let maxy = maximum ys in trace (show maxy) fixedScaleAxis Linear chartH 0 $ maxy


mychart2 pairs = 
    pictures [ color bgColor $ plotChartBackground xAxis yAxis
             , color gridColor $ plotGrid xAxis yAxis (0, 0.125)
             , plotAxes xAxis yAxis
             , plotAxisScales xAxis yAxis (2, 0.5)
             , line 1 pointColor
             , line 1.5 pointColor'
             , line 3 pointColor''
             ]
    where xAxis     = autoScaleAxis Linear chartW xs
          yAxis     = fixedScaleAxis Linear chartH 0 10
          (xs,ys)        = unzip pairs
          pts x     = pairs
          line x c  = color c $ plotLineChart xAxis yAxis (pts x)
          sigmoid x = 1.0 / (1 + exp (-x))
          
lineChart :: Picture
lineChart = 
    pictures [ color bgColor $ plotChartBackground xAxis yAxis
             , color gridColor $ plotGrid xAxis yAxis (0, 0.125)
             , plotAxes xAxis yAxis
             , plotAxisScales xAxis yAxis (2, 0.5)
             , line 1 pointColor
             , line 1.5 pointColor'
             , line 3 pointColor''
             ]
    where xAxis     = autoScaleAxis Linear chartW xs
          yAxis     = fixedScaleAxis Linear chartH 0 1
          xs        = [-6,-5.75..6]
          ys        = [sigmoid x | x <- xs]
          pts x     = zip xs (map ((+(0.5 - sigmoid 0 / x)) . (/x)) ys)
          line x c  = color c $ plotLineChart xAxis yAxis (pts x)
          sigmoid x = 1.0 / (1 + exp (-x))
          
          
-----------------------------------------------------------------------------
logScaleChart :: Picture
logScaleChart =
    pictures [ color bgColor $ plotChartBackground xAxis yAxis
             , plotAxes xAxis yAxis
             , color pointColor $ plotAxisScales xAxis yAxis (1, 10)
             , translate (chartW + 25) 0 $ color pointColor' 
                                         $ plotAxisScales xAxis yAxis' (0, 10)
             , color pointColor $ plotLineChart xAxis yAxis $ zip xs ys
             , color pointColor' $ plotLineChart xAxis yAxis' $ zip xs ys
             ]
    where xAxis     = fixedScaleAxis Linear chartW 0 10
          yAxis     = fixedScaleAxis Linear chartH 0 1024
          yAxis'    = fixedScaleAxis Log chartH 1 1000
          xs        = [0.1, 0.25..10]
          ys        = map (2 **) xs
