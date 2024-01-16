import requests 
import json 
import os
import logging
from datetime import datetime
import pandas as pd 
#API_KEY with read permissions 
#Written as ENV_VAR in Render


def get_keys() -> dict:
    '''
    Gets relevant 

    kwargs: None
    returns: key_dict : dict =  {API_OKX_PUBLIC : str,
                                 API_OKX_SECRET : str,
                                 API_S3_ID : str,
                                 API_S3_SECRET : str}
    '''
    _log = logging.getLogger(__name__)
    _log.info("Getting keys...")
    #Get vars
    API_OKX_PUBLIC = os.getenv("API_OKX_PUBLIC")
    API_OKX_SECRET = os.getenv("API_OKX_SECRET")
    API_S3_ID = os.getenv("API_S3_ID")
    API_S3_SECRET = os.getenv("API_S3_SECRET")
    #Get dict
    key_dict = {"API_OKX_PUBLIC" : API_OKX_PUBLIC, 
                "API_OKX_SECRET" : API_OKX_SECRET, 
                "API_S3_ID" : API_S3_ID,
                "API_S3_SECRET" : API_S3_SECRET}
    
    #Validating vars
    if None not in key_dict.keys():
        _log.info("Succesfully got keys!")
    else:
        raise ValueError("Env vars not stored properly.")
    
    return key_dict

def get_tickers() -> list:
    '''
    Wraps /api/v5/market/tickers/

    Get list of valid SPOT market tickers.

    kwargs: None 

    returns: tickers : list
    '''
    _log = logging.getLogger(__name__)
    _log.info("Getting currently quoted SPOT tickers...")
    base_url = 'https://www.okx.com'
    url = 'https://www.okx.com'
    tickers = pd.DataFrame((requests.get(url+'/api/v5/market/tickers?instType=SPOT').json())['data'])
    tickers = tickers.drop('instType', axis=1)
    tickers = tickers.instId
    _log.info("Tickers retrieved.")
    return list(tickers)

    
def get_history_by_ticker(ticker : str, timeframe : str, num_bars_back : int = 300):
    '''
    Wraps OKX API GET / Candlesticks history method. 
    Counts number of bars back (since the limit is 100) and concatenates
    several requests into a single array. The default is 300.

    Please note that crypto markets work 24/7 hence the logic for gathering all relevant data.
    
    Checks for ticker input validity first, althoug with no suggestions.
    
    kwargs: ticker : str, ticker of an asset, SPOT
            timeframe : str, bar_timeframe e.g. "5m", "1h" etc 
            num_bars_back : int, number of bars back to get data.

    returns: ohlc : Union[list, np.array, pd.DataFrame, pd.Series, dict]
    '''
    
    #Historical data: https://www.okx.com/api/v5/market/history-candles?instId=BTC-USD-190927
    #Take a date long in the past e.g. 190927 : YYMMDD
    _log = logging.getLogger(__name__)
    _log.info("Starting to GET data...")
    #Check for ticker validity
    _log.info("Validating ticker...")
    valid_tickers = get_tickers()
    if ticker in valid_tickers:
        pass
    else:
        raise ValueError(f"{ticker} is not a valid ticker.")
        
    response_cols = ["unix_time", "open", "high", "low", "close", "volume", "volumeCurrency","volCcyQuote", "closed_flg"]
    base_url = 'https://www.okx.com'
    method = 'api/v5/market/history-candles'
    payload = {'InstId': ticker, 'bar': timeframe, 'limit' : num_bars_back}
    
    assert num_bars_back > 0
    
    _log.info(f"Getting {num_bars_back} {timeframe} bars OHLC data for {ticker}")
    
    if num_bars_back <= 300:
        #300 is single-call limit set by provider (OKX)
        _log.info(f"Making a request to {method}")
        
        response = requests.get(f"{base_url}/{method}", params = payload)
        _log.info(f"Status code {response.status_code} returned")
        response_data = response.json()["data"]
        
        ohlc = pd.DataFrame(response_data, columns = response_cols)
        _log.info(f"Data downloaded!")
        
    else:
        #Lazy logic implemented, may return more data than it should. May ask for unexisting data in some cases.
        #To be fixed later. 
        _log.info(f"Making a request to {method}")
        
        response = requests.get(f"{base_url}/{method}", params = payload)
        
        _log.info(f"Status code {response.status_code} returned")
        response_data = response.json()["data"]

        if (num_bars_back // 300) * 300 < num_bars_back:
            num_batches = num_bars_back // 300 + 1
        else:
            num_batches = num_bars_back // 300
        
        ohlc = pd.DataFrame(response_data, columns = response_cols)
        
        cur_num_back = num_bars_back - 300
        last_cur_timestamp = ohlc.unix_time[-1]
        counter = 1
        _log.info(f"Batch {counter} out of {num_batches} downloaded!")
        
        while cur_num_back >= 0:
            payload["after"] = last_cur_timestamp
            _log.info(f"Making a request to {method}")
            response = requests.get(f"{base_url}/{method}", params = payload)
            _log.info(f"Status code {response.status_code} returned")
            
            response_data = response.json()["data"]
            
            new_ohlc = pd.DataFrame(response_data, columns = response_cols)

            ohlc = pd.concat([ohlc, new_ohlc])

            counter += 1
            _log.info(f"Batch {counter} out of {num_batches} downloaded!")
            
            last_cur_timestamp = ohlc.unix_time[-1]
            cur_num_back += -300 
            
    return ohlc


if "name" == "__main__":
    _log = logging.getLogger(__name__)
    _log.info("Starting to test functionality...")
    _log.info("Testing key retrieval...")
    key_dict = get_keys()
    
    API_OKX_PUBLIC = key_dict["API_OKX_PUBLIC"]
    API_OKX_SECRET = key_dict["API_OKX_SECRET"]
    
    _log.info("Testing ticker retrieval...")
    tickers = get_tickers()
    
    test_ticker = tickers[-42]
    test_tf = "5m"
    num_bars_back = 302 # >300
    _log.info("Testing getting data retrieval...")
    ohlc = get_history_by_ticker(test_ticker, test_tf, num_bars_back)
    
    _log.info("Implemented tests succesful!")
    
    
    

