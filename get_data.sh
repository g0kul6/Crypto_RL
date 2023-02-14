##!/bin/bash
# echo "-------------------------------------------------------"
# echo "-------------------------------------------------------"
# echo "Available intervals (choose any one from below) :"
# echo """->1m
# ->2m
# ->5m
# ->15m
# ->30m
# ->60m
# ->90m
# ->1h
# ->1d
# ->5d
# ->1wk
# ->1mo
# ->3mo"""
# echo "-------------------------------------------------------"
# echo "-------------------------------------------------------"
# echo "Available period (choose any one from below) :"
# echo """->1d (1 Day historical data from now if applicable)
# ->5d (5 Dayshistorical data from now if applicable)
# ->1mo (1 Month historical data from now if applicable)
# ->3mo (3 Months historical data from now if applicable)
# ->6mo (6 Months historical data from now if applicable)
# ->1y (1 Year historical data from now if applicable)
# ->2y (2 Years historical data from now if applicable)
# ->5y (5 Years historical data from now if applicable)
# ->10y (10 Years historical data from now if applicable)
# ->ytd (Year to date historical data from now)
# ->max (Maximum historical data)"""
# echo "-------------------------------------------------------"
# echo "-------------------------------------------------------"

# echo "enter coin_id"
# read a

# echo "enter inteval"
# read i

# echo "enter period"
# read p

# python get_data.py --coin_id $a --interval $i --period $p 