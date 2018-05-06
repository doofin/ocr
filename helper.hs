module Main where

import Graphics.HsCharts
import Graphics.Gloss
import Text.Parsec.Prim
import Text.Parsec.Combinator
import Text.Parsec.Char
import Text.Parsec.Token
import Debug.Trace
import Text.Parsec.String
import System.Cmd
import System.Process

ps str = case (parse tpps "" (str :: String)) of
  Left err  -> trace (show err) []
  Right xs  -> xs

tpps::Parser [[Float]]
tpps = lne `endBy` (char '\n')

lne::Parser [Float]
lne = (fmap (read :: String->Float) (many (digit <|> char '.'))) `sepBy` (char ',' )

 
-- if you parse a stream of characters
biniarize = system "convert pp2.png -threshold 40% thres_colored2.png"
concatPdf = system "pdfunite front.pdf thesisnew.pdf o.pdf" 
 
main = concatPdf
