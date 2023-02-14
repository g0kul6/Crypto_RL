# from matplotlib.pyplot import step
# import yfinance as yf
# import argparse
# import bitfinex
# import datetime
# import time

# # parser = argparse.ArgumentParser(description='Data id,interval and period')
# # parser.add_argument('--coin_id', type=str, required=True)
# # parser.add_argument('--interval', type=str, required=True)
# # parser.add_argument('--period', type=str, required=True)
# # args = parser.parse_args()

# # i=args.interval
# # p=args.period
# # coin_id=args.coin_id

# # ticker = yf.Ticker(coin_id)
# # data = ticker.history(interval=i,period=p)
# # data = data.to_csv("/home/g0kul6/g0kul6/RL-PROJECT/Project_RL/dataset/BITCOIN/{}_{}_intevral_{}_period.csv".format(coin_id,i,p))
# # # print(BTC_Data)

# api_v2 = bitfinex.bitfinex_v2.api_v2()
# pair="btcusd"
# bin_size="1m"
# time_step = 60000000

# t_start=datetime.datetime(2013, 4, 1, 0, 0)
# t_start = time.mktime(t_start.timetuple()) * 1000

# t_end=datetime.datetime(2022, 4, 1, 0, 0)
# t_end = time.mktime(t_end.timetuple()) * 1000

