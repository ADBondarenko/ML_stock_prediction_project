import pandas as pd 
import numpy as np
import logging
import re 
from sklearn.model_selection import train_test_split

def gen_re_rsi(data: pd.DataFrame, level : int = 50, period : int = 14):
    '''
    Reverse Engineered RSI for a single period based on 
    https://www.tradingview.com/script/Di9aEill-Reverse-Engineering-RSI-by-Giorgos-Siligardos/
    which is in turn an implementation of 2003 article in 
    Journal of Techincal Analysis of Stocks and Commodities.
    Implementation is for a single level 
    
    kwargs: data : pd.DataFrame with columns:
                    ["unix_time", "open", "high", "low", 
                    "close", "volume", "volumeCurrency",
                    "volCcyQuote", "closed_flg"]

    returns: re_rsi : list
    '''
    _log = logging.getLogger(__name__)
    
    close_prices = data.close
    ExpPer = 2 * period - 1
    K = 2 / (ExpPer + 1)
    
    AUC = 0.0
    ADC = 0.0

    nVal = np.zeros_like(close_prices)
    nRes = np.zeros_like(close_prices)
    _log.info(f"Generating RE-RSI features for {level} level")
    for i in range(1, len(close_prices)):
        if close_prices[i] > close_prices[i - 1]:
            AUC = K * (close_prices[i] - close_prices[i - 1]) + (1 - K) * (AUC if i > 1 else 1)
            ADC = (1 - K) * (ADC if i > 1 else 1)
        else:
            ADC = K * (close_prices[i - 1] - close_prices[i]) + (1 - K) * (ADC if i > 1 else 1)
            AUC = (1 - K) * (AUC if i > 1 else 1)

        nVal[i] = (period - 1) * (ADC * level / (100 - level) - AUC)
        nRes[i] = close_prices[i] + nVal[i] if nVal[i] >= 0 else close_prices[i] + nVal[i] * (100 - level) / level
    _log.info(f"Generation succesfull")
    
    re_rsi = nRes
    
    return re_rsi
    
def gen_re_rsi_mtf(data: pd.DataFrame, levels : list = [20,40,60,80], period : int = 14):
    '''
    Calls gen_re_rsi() for multiple levels for a single sampling period.
    Returns a dict of levels and their respective values.
    
    kwargs: data : pd.DataFrame with columns:
                    ["unix_time", "open", "high", "low", 
                    "close", "volume", "volumeCurrency",
                    "volCcyQuote", "closed_flg"]
            levels : list = [20, 40, 60, 80], a list of relevant RSI levels
            period : int = 14, sampling period

    returns: re_rsi_mtf : dict
    '''
    _log = logging.getLogger(__name__)
    close_prices = data.close
    
    assert len(close_prices) > period
    
    re_rsi_mtf = {}
    _log.info(f"Generating RE-RSI features for {levels} levels")
    for level in levels: 
        re_rsi_mtf[f"re_rsi_{level}_{period}"] = gen_re_rsi(data, level, period)
    _log.info(f"Generation succesfull")

    return re_rsi_mtf
        
    

def gen_rsi(data : pd.DataFrame, period : int = 14):
    '''
    Generate RSI for close prices. Returns an np.ndrray of a given feature.
    kwargs: data : pd.DataFrame with columns:
                    ["unix_time", "open", "high", "low", 
                    "close", "volume", "volumeCurrency",
                    "volCcyQuote", "closed_flg"]
            period : int = 14, sampling period

    returns: rsi_values_dict : dict
    '''
    close_prices = data.close
    price_changes = np.diff(close_prices)
    _log = logging.getLogger(__name__)
    assert len(close_prices) > period
    _log.info(f"Generating RSI({period}) feature...")
    rsi_values = []
    
    for i in range(0,len(close_prices)):
        if i <= period:
            rsi_values.append(np.nan)
        else:
            avg_gain = np.mean(np.where(price_changes[i-period:i] < 0, price_changes[i-period:i], 0))
            avg_loss = -np.mean(np.where(price_changes[i-period:i] > 0, price_changes[i-period:i], 0))
    
            rs = avg_gain / avg_loss 
            rsi = 100 - (100 / (1 + rs))
    
            rsi_values.append(rsi)
        
    _log.info(f"RSI({period}) feature generated")

    rsi_values_dict = {}
    rsi_values_dict[f"rsi_{period}"] = rsi_values
    
    return rsi_values_dict

def gen_ema(data : pd.DataFrame, period : int = 8):
    '''
    Generates an exponentially decaying moving average for a given 
    set of close prices.
    
    kwargs: data : pd.DataFrame with columns:
                    ["unix_time", "open", "high", "low", 
                    "close", "volume", "volumeCurrency",
                    "volCcyQuote", "closed_flg"]
            period : int = 8, sampling period    
            
    returns: ema_values_dict : dict
    '''
    _log = logging.getLogger(__name__)
    values = data.close
    assert len(values) > period 
    _log.info(f"Generating EMA({period}) feature...")
    
    alpha = 2 / (period  + 1)
    ema_values = [values[0]]  

    for i in range(1, len(values)):
        ema = alpha * values[i] + (1 - alpha) * ema_values[-1]
        ema_values.append(ema)

    ema_values_dict = {}
    ema_values_dict[f"ema_{period}"] = ema_values
    
    return ema_values_dict


def gen_binary_features(data : pd.DataFrame, EMA : bool = True, RSI : bool = True):
    '''
    Accepts a CROPPED by a period factor 
    pd.DataFrame with RSI/EMA features already generated. If binary flags are activated,
    returns respective features. It is a binary flag for respective 
    EMA and ordinal data for RSI.

    Checks that the columns pass the naming convention i.e. 
    rsi_re_%{level} is in column names for RSI features, ema_%{period} for EMA features.
    Returns a dictionary of binary features.
    
        kwargs: data : pd.DataFrame with columns:
                    ["unix_time", "open", "high", "low", 
                    "close", "volume", "volumeCurrency",
                    "volCcyQuote", "closed_flg"]
                EMA : bool = True, whether to generate EMA features flag
                RSI : bool = True, whether to generate EMA features flag
                
    returns: rsi_ema_flg_dict : dict
    '''
    _log = logging.getLogger(__name__)
    columns = data.columns
    close_prices = data.close
    
    rsi_ema_flg_dict = {}
    if EMA == True:
        for col_name in columns:
            if "ema" in col_name:
                period = col_name.split('_')[1]
                ema_vals = data[[col_name]]
                flag_array =[]
                for closing_price, ema_val in zip(close_prices, ema_vals.values):
                    if closing_price > ema_val:
                        flag_array.append(1)
                    
                    else:
                        flag_array.append(0)

                rsi_ema_flg_dict[f"EMA_{period}_flg"] = flag_array
    
    if RSI == True:
        for col_name in columns:
            if ("rsi" in col_name) and ("re" not in col_name):
                period = col_name.split('_')[1]
                rsi = data[col_name]
        levels = []
        for col_name in columns:
            if "rsi_re" in col_name:
                level = col_name.split("_")[2]
                level.append(level)
        num_levels = len(levels)

        rsi_ord = []
        for rsi, closing_price in zip(rsi, close_prices):
            ordinal_level = 0
            for level in levels:
                if rsi >= level:
                    ordinal_level += 1
            rsi_ord.append(ordinal_level)
                
        rsi_ema_flg_dict["rsi_ordinal"] = rsi_ord

    if (EMA or RSI) == True:
        return rsi_ema_flg_dict
    else: 
        return None

def gen_nth_diff(data : pd.DataFrame, n : int = 1):
    '''
    Receives a pd.DataFrame with columns:
    ["unix_time", "open", "high", "low", "close", "volume", "volumeCurrency","volCcyQuote", "closed_flg"]
    generates n-th difference for open, high, low, close columns.

    kwargs: data : pd.DataFrame with columns:
                    ["unix_time", "open", "high", "low", 
                    "close", "volume", "volumeCurrency",
                    "volCcyQuote", "closed_flg"]
            n : int = 1, number of differences back
            
    returns: diff_dict : dict, a dictionary of differences
    '''
    _log = logging.getLogger(__name__)
    assert n >= 1
    diff_dict = {}
    for nth in range(1, n+1):
        diff_dict[f"close_{nth}_diff"] = data.close.diff(nth)
        diff_dict[f"open_{nth}_diff"] = data.open.diff(nth)
        diff_dict[f"high_{nth}_diff"] = data.high.diff(nth)
        diff_dict[f"low_{nth}_diff"] = data.low.diff(nth)

    return diff_dict


    

def generate_features(data : pd.DataFrame, binary_rsi : bool = True, binary_ema : bool = True, nth_diff : int = 1,
                      rsi_period : int = 14, rsi_levels : list = [20,40,60,80], 
                      ema_periods : list = [8,24]):
    '''
    Receives a pd.DataFrame with columns:
    ["unix_time", "open", "high", "low", "close", "volume", "volumeCurrency","volCcyQuote", "closed_flg"]

    Generate features such as 
    countinous variables: 
    RSI, Reverse engineered RSI for 20/40/50/60/80 RSI levels, EMA with 8/24 periods, 
    
    binary variables (may introduce multicollinearity hence the feature is optional:
    flag of being in between certain RSI levels, above or below EMA8/EMA24

    returns: enriched_data : pd.DataFrame with generated features.
    '''
    _log = logging.getLogger(__name__)
    #Generate RSI
    rsi = gen_rsi(data, rsi_period)
    rsi_key = list(rsi.keys())[0]
    data[rsi_key] = rsi[rsi_key]
    #Generate RE_RSI
    re_rsi_mtf = gen_re_rsi_mtf(data, rsi_levels, rsi_period)

    for key in re_rsi_mtf:
        data[key] = re_rsi_mtf[key]
    #Generate EMA
    for period in ema_periods:
        ema = gen_ema(data, period)
        ema_key = list(ema.keys())[0]
        data[ema_key] = ema[ema_key]
        
    binary = (binary_rsi or binary_ema)
    #Generate binary and ordinal  data
    if binary == True:
        binary_features = gen_binary_features(data, binary_ema, binary_rsi)
        for key in binary_features.keys():
            data[key] = binary_features[key]
    #Generate n-th differences
    differences = gen_nth_diff(data, nth_diff)
    for key in differences.keys():
        data[key] = differences[key]
        
    #Add a previous OHLC
    data_prev_cols = ["open_1_back",
                       "close_1_back", 
                       "low_1_back",
                       "high_1_back"]
    
    data[data_prev_cols] = data[["open", "close", "low", "high"]].shift(1)
        
    return data        
        
def cleanup_and_prepare_data(data : pd.DataFrame):
    _log = logging.getLogger(__name__)
    #Dropping duplicates for SPOT markets
    data = data.drop(columns = ["volumeCurrency","volCcyQuote"])
    
    #Dropping open candles
    open_candle_ind = data.loc[data.closed_flg == 0].index
    data = data.drop(index = open_candle_ind)

    #Setting time_index
    data = data.set_index("unix_time")

    
    #Creating a target - next closing price
    data["target"] = data.close.shift(-1)
    
    #Cropping_data  
    shape_now = data.shape
    _log.info(f"Start dropping rows containing nans...\
                The shape is {shape_now}")
    data = data.dropna()
    shape_then = data.shape
    _log.info(f"Dropped rows containing nans.\
                The shape is {shape_then}")
    X, X_col = data.drop(columns = "target"), data.drop(columns = "target").columns
    y = data.target
    X_train, X_val, y_train, y_val = train_test_split(X,y test_size = 0.2, shuffle = False), 

    return [X_train, y_train, X_val, y_val], X_col
    
    
    

    
    
    
    