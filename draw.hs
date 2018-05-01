module Main where

import Graphics.HsCharts
import Graphics.Gloss
import Text.Parsec.Prim
import Text.Parsec.Combinator
import Text.Parsec.Char
import Text.Parsec.Token
import Debug.Trace
import Text.Parsec.String


ps str = case (parse tpps "" (str :: String)) of
  Left err  -> trace (show err) []
  Right xs  -> xs

tpps::Parser [[Float]]
tpps = lne `endBy` (char '\n')

lne::Parser [Float]
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
      q 0 0 $ mychartLin pairsv,
      q 1 0 $ mychartLin pairst,
      q 2 0 $ mychartLog pairsv,
      q 3 0 $ mychartLog pairst

      ]
    
windowW = 1800
windowH = 1000
chartW  = fromIntegral $  windowW 
chartH  = fromIntegral $ windowH 
chartM  = 50
drawInterval=10000
q x y  = translate dx dy
    where dx = (-(fromIntegral windowW / 2)) + (chartM + x * (chartW + chartM)) + 200
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

mychartLin pairs = 
    pictures [ color bgColor $ plotChartBackground xAxis yAxis
             , color gridColor $ plotGrid xAxis yAxis (1000, 0.1)
             , plotAxes xAxis yAxis
             , plotAxisScalesSZ 0.2 xAxis yAxis (drawInterval, 0.2)
             , line  pointColor
             , line  pointColor'
             , line  pointColor''
             ]
    where xAxis     = autoScaleAxis Linear chartW xs
          yAxis     = fixedScaleAxis Linear chartH 0 3
          (xs,ys)        = unzip pairs
          line c  = color c $ plotLineChart xAxis yAxis pairs
          
mychartLog pairs = 
    pictures [ color bgColor $ plotChartBackground xAxis yAxis
             , color gridColor $ plotGrid xAxis yAxis (1000, 0.1)
             , plotAxes xAxis yAxis
             , plotAxisScalesSZ 0.2 xAxis yAxis (drawInterval, 0.2)
             , line  pointColor
             , line  pointColor'
             , line  pointColor''
             ]
    where xAxis     = autoScaleAxis Linear chartW xs
          yAxis     = fixedScaleAxis Log chartH 0.1 3
          (xs,ys)        = unzip pairs
          line c  = color c $ plotLineChart xAxis yAxis pairs

          
