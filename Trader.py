#!/usr/bin/env python3.6
# coding: utf-8 
# # Import

# In[1]:


import csv,importlib
import pandas as pd
import numpy as np
import matplotlib as mpl
#%matplotlib inline #not used in scripts
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import urllib.request
import json
from pandas.io.json import json_normalize
import glob
import plotly.plotly as py
import plotly.graph_objs as go
from bokeh.plotting import figure, show, output_file, save
from bokeh.layouts import column
from bokeh.layouts import row
from bokeh.embed import file_html
from bokeh.resources import CDN
from bokeh.embed import components
from math import pi,sqrt
from bokeh.models import HoverTool
import datetime
from scipy.optimize import curve_fit
from uncertainties import ufloat
import ftplib


# # Functions

# ## General

# In[2]:


def filejson2dictionary(fn):
    with open(fn) as json_data:
        d = json.load(json_data)
    return d


# In[3]:


def publish_ftp_file(filename):
    login=filejson2dictionary('login.json')
    user=login['username']
    pswd=login['password']
    hostname=login['hostname']
    session = ftplib.FTP(hostname,user,pswd)
    file = open(filename,'rb')                      # file to send
    print( ftplib.FTP.pwd(session) )
    ftplib.FTP.cwd(session,'/public_html')
    session.storbinary('STOR '+filename, file)      # send the file
    file.close()                                    # close file and FTP
    session.quit()


# ## Crypto

# ### Coincap

# In[4]:


def get_symbol(ETH,_property,days=365):
    url = 'http://coincap.io/history/'+str(days)+'day/'+ETH

    req = urllib.request.Request(
        url, 
        data=None, 
        headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
        }
    )

    response = urllib.request.urlopen(req)
    obj = json.load(response)
    df = pd.DataFrame(obj[_property], columns=['EpochTime', _property])
    df['time'] = pd.to_datetime(df['EpochTime'], unit='ms')
    df['date']=df['time']#.dt.date
    res=pd.DataFrame(obj[_property],index=df['date'],columns=['EpochTime',_property])
    return res


# ### Binance

# In[5]:


def get_symbol_bnb(ETH='ADABTC'):
    url = 'https://api.binance.com//api/v1/klines?symbol='+str(ETH)+'&interval=5m'
    req = urllib.request.Request(url, data=None, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'})
    response = urllib.request.urlopen(req)
    obj = json.load(response)
    candel_stikcs=pd.DataFrame(obj,columns=['OpeningTime','Open','High','Low','Close','Volume','ClosingTime','QuoteAssetValue','NumberOfTrades','TakerBuyBaseAssetVolume','TakerBuyQuoteAssetVolume','Ignore'])
    candel_stikcs["Date"] = pd.to_datetime(candel_stikcs["OpeningTime"],unit='ms')
    return candel_stikcs

#/api/v1/klines


# ### GDAX

# In[6]:


def get_trades(ETH,EUR):
    url = 'https://api.gdax.com/products/'+str(ETH)+'-'+str(EUR)+'/trades'

    req = urllib.request.Request(
        url, 
        data=None, 
        headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
        }
    )

    response = urllib.request.urlopen(req)
    obj = json.load(response)
    #df = pd.DataFrame(obj[_property], columns=['EpochTime', _property])
    #df['time'] = pd.to_datetime(df['EpochTime'], unit='ms')
    #df['date']=df['time']#.dt.date
    #res=pd.DataFrame(obj[_property],index=df['date'],columns=['EpochTime',_property])
    return obj


# In[24]:


def get_symbol_gdx(ETH,minutes=5,fiat='EUR',points=20,endtime='now'):
    if minutes in [1,5,15,60,360,1440]:
        if endtime == 'now':
            end=datetime.datetime.utcnow() #else is assumed to be a datetime.datetime
        else:
            end=endtime
        start = end - datetime.timedelta(minutes=points*minutes)
        _start = str(start.isoformat())+'Z'
        _end=str(end.isoformat())+'Z'
        print(_start,' --->',_end)
        #one minute, five minutes, fifteen minutes, one hour, six hours, and one day, respectively.
        url = 'https://api.gdax.com/products/'+ETH+'-'+fiat+'/candles?granularity='+str(60*minutes)+'&start='+_start+'&end='+_end
        #print(url)
        req = urllib.request.Request(url, data=None, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'})
        response = urllib.request.urlopen(req)
        obj = json.load(response)
        _candel_stikcs=pd.DataFrame(obj,columns=['OpeningTime','Low','High','Open','Close','Volume'])
        #print(response)
        #print(len(_candel_stikcs),len(obj))
        #time, low, high, open, close, volume
        _candel_stikcs["Date"] = pd.to_datetime(_candel_stikcs["OpeningTime"],unit='s')
        
        _filtered = _candel_stikcs[(_candel_stikcs['Date'] > _start )] #& (df['date'] < '2013-02-01')]
       
        return _filtered
    else:
        print(minutes,' is not valid')
#/api/v1/klines


# In[25]:


get_symbol_gdx('BTC')


# In[8]:


def get_symbol_weekly(token,price):
    _token_price=get_symbol(token,price)
    token_price=_token_price.resample('W',label='left', closed='left').mean()
    return token_price


# In[9]:


def get_symbol_nhours(token,price):
    _token_price=get_symbol(token,price,days=1)
    token_price=_token_price#.resample('W',label='left', closed='left').mean()
    return token_price


# ### Google Trends Analysis

# In[10]:


def get_searches(term):
    searches = pd.read_csv('./'+term+'.csv', index_col=['Week'], names=['Week', 'Searches'],skiprows=None,header=1)
    return searches


# In[11]:


def sideBySide(term,token):
    _newdfdiff=get_searches(term).pct_change(periods=1)
    _l=get_symbol_weekly(token,'price').head(-1).pct_change(periods=1)
    #print(_l)
    #print(len(_l),len(_newdfdiff))
    _newdfdiff['price']=_l['price']
    return _newdfdiff


# In[12]:


def correlation(term,symbol):
    sbs=sideBySide(term,symbol)
    return sbs['Searches'].corr(sbs['price'])


# ### CandelSticks Functions

# In[13]:


def risk_matrix(newdfdiff,abshorObs='Volume',absvertObs='MidMarket'):
    
    _range=np.array([-1,-0.3,-0.1,0.1,0.3,1,2,3,5])
    horObs='pct'+abshorObs
    vertObs='pct'+absvertObs
    
    newdfdiff[vertObs]=newdfdiff[absvertObs].pct_change(periods=1)
    newdfdiff[horObs]=newdfdiff[abshorObs].pct_change(periods=1)

    print("rho=",newdfdiff[horObs].corr(newdfdiff[vertObs]) )
    
    _C=newdfdiff.groupby(pd.cut(newdfdiff[horObs], _range)).count()
    _M=newdfdiff.groupby(pd.cut(newdfdiff[horObs], _range)).mean()
    _S=newdfdiff.groupby(pd.cut(newdfdiff[horObs], _range)).std()
    __min=newdfdiff.groupby(pd.cut(newdfdiff[horObs], _range)).min()
    __max=newdfdiff.groupby(pd.cut(newdfdiff[horObs], _range)).max()
    _C['counts']=_C[vertObs]
    _C['mean']=_M[vertObs]
    _C['std']=_S[vertObs]
    _C['min']=__min[vertObs]
    _C['max']=__max[vertObs]
    #risk_matrix(candel_stikcs)
    return _C[['pctVolume','MidMarket','counts','mean','std','min','max']]
    


# In[14]:


def candel_frequency(cs):
    candel_stikcs=cs.copy()        
    if isinstance(cs, pd.DataFrame):
        candel_stikcs['delta'] = (candel_stikcs['Date'].shift()-candel_stikcs['Date']).fillna(0)
    if isinstance(cs, pd.Series):
        candel_stikcs['delta'] = (cs.shift()-cs).fillna(0)

    diffs=candel_stikcs['delta'][1:]
    #return float(diffs.mean().total_seconds())
    return float(diffs.min().total_seconds())


# In[15]:


def fit_periods(_candel_stikcs,High='High',Low='Low',units='EUR'):
    
    _candel_stikcs['Spread']=abs(_candel_stikcs[High]-_candel_stikcs[Low])
    _candel_stikcs['MidMarket']=(_candel_stikcs[High]+_candel_stikcs[Low]).divide(2.0)
    #candel_stikcs=_candel_stikcs[_candel_stikcs['Spread']!=0].copy()
    candel_stikcs=_candel_stikcs.copy()
    _weights=candel_stikcs['Spread']#.apply(sqrt)
    if ((candel_stikcs[High]==candel_stikcs[Low]).min() == (candel_stikcs[High]==candel_stikcs[Low]).max()):
        _weights=None 
        #print('dropping weights vector in the fit, going with None')
    #
    candel_stikcs['TimeFromT0']=(candel_stikcs['OpeningTime']-candel_stikcs['OpeningTime'][len(candel_stikcs)-1]).divide(60)
    _x=candel_stikcs['TimeFromT0']
    #
    _y=candel_stikcs['MidMarket']#(candel_stikcs[High]+candel_stikcs[Low]).divide(2.0)
    
    def _func(x, a, b):
        return a + b * x

    popt, pcov = curve_fit(_func, _x, _y,sigma=_weights,p0=[_y.max(),0])
    #print("({:.2g}".format(popt[0]),'±',"{:.2g})".format(sqrt(pcov[0,0])) ,'+', "({:.2g}".format(popt[1]),"±","{:.2g}".format(sqrt(pcov[1,1])),")* t/min" )
    #print("({:.2g}".format(popt[0]),'±',"{:.2g})".format(sqrt(pcov[0,0])) ,'+', "({:.2g}".format(60*popt[1]),"±","{:.2g}".format(60*sqrt(pcov[1,1])),")* t/h" )
    _const=ufloat(popt[0],sqrt(pcov[0,0]))
    _linear=ufloat(popt[1],sqrt(pcov[1,1]))
    print('{:+.1uS}'.format(_const) ,units+'+', '( {:+.1uS}'.format(60*_linear)," )"+units+"* t/h" )
    #
    lin=ufloat(popt[1],sqrt(pcov[1,1]))
    const=ufloat(popt[0],sqrt(pcov[0,0]))
    #
    return lambda x:_func(x,popt[0],popt[1]), _const, _linear*60/_const#popt[1]*60/popt[0]


# In[16]:


def make_candle_plot(df,title='Candles',plot_width=1000,normalized=False,one=-1,High='High',Low='Low',html5_components=False):
    #print(df)
    Close='Close'
    Open='Open'
    High='High'
    Low='Low'
    if normalized:
        norm=df['Open'].iloc[one]
        df['nOpen']=df['Open'].astype('float').divide(float(norm))
        df['nClose']=df['Close'].astype('float').divide(float(norm))
        df['nHigh']=df['High'].astype('float').divide(float(norm))
        df['nLow']=df['Low'].astype('float').divide(float(norm))
        #
        Close='nClose'
        Open='nOpen'
        High='nHigh'
        Low='nLow'
    inc = df[Close] > df[Open]
    dec = df[Open] > df[Close]
    #w = 12*60*60*1000 # half day in ms
    w = 500*500*candel_frequency(df)/300
    #print('frequency:',candel_frequency(df)/60,' mins') 
    TOOLS = "pan,wheel_zoom,box_zoom,reset,save"
    start_time=make_time_of_plot(df['Date'].loc[0]) 
    cumtitle=title+" "+start_time
    
    if not normalized:
        _F360, constant, der =fit_periods(df,High=High,Low=Low,units='EUR')
        points=len(df.Date)
        mins=candel_frequency(df)/60
        X=np.arange(0,mins*points,mins)
        #
        #print('{:+.1uS}'.format(_const) ,units+'+', '({:+.1uS}'.format(60*_linear),")"+units+"* t/h" )
        #cumtitle=cumtitle+" ({:.2g}".format(der*100)+")%/h"
        cumtitle=cumtitle+" [ {:+.1uS}".format(der*100)+" ]%/h"
    
    p = figure(x_axis_type="datetime", tools=TOOLS, plot_height=plot_width,plot_width=plot_width, title = cumtitle,x_axis_label='Time',y_axis_label='Price')
    p.xaxis.major_label_orientation = pi/4
    p.xaxis.axis_label_text_font_size = "8pt"
    p.yaxis.axis_label_text_font_size = "8pt"
    p.title.text_font_size = '8pt'
    p.grid.grid_line_alpha=0.3

    p.segment(df.Date, df[High], df.Date, df[Low], color="black")
    p.vbar(df.Date[inc], w, df[Open].loc[inc], df[Close].loc[inc], fill_color="#D5E1DD", line_color="black")
    p.vbar(df.Date[dec], w, df[Open].loc[dec], df[Close].loc[dec], fill_color="#F2583E", line_color="black")
    
    if not normalized:
        p.line(df.Date.iloc[::-1],_F360(X), line_width=2)

    if not html5_components:
        return p
    if html5_components:
        scr, div = components(p)
        return p, div, scr


# ### Bokeh

# In[17]:


def scrips_from_components(result):
    _scripts=''
    for res in result:
        for plotdata in res:
            _scripts=_scripts+plotdata[2]
    return _scripts
        
def body_from_components(result):
    _body=''
    i=0
    for plot in result:
        _bodycol=''
        #print(i)
        for plotdata in plot:
            _bodycol=_bodycol+html5_row(plotdata[1])
        _body=_body+html5_column(_bodycol)   
    return _body    


def HTMLfromScriptsBody(_scripts,_body,headline):
    
    _head = '<meta charset="utf-8">    <link rel="stylesheet" href="https://cdn.pydata.org/bokeh/release/bokeh-0.12.13.min.css" type="text/css">    <script type="text/javascript" src="https://cdn.pydata.org/bokeh/release/bokeh-0.12.13.min.js"></script>    <script type="text/javascript">      Bokeh.set_log_level("info");    </script>    <meta name="viewport" content="width=device-width, initial-scale=1">    <script type="text/javascript" src="http://cdnjs.cloudflare.com/ajax/libs/jquery/2.0.3/jquery.min.js"></script>    <script type="text/javascript" src="http://netdna.bootstrapcdn.com/bootstrap/3.3.4/js/bootstrap.min.js"></script>    <link href="http://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.3.0/css/font-awesome.min.css"    rel="stylesheet" type="text/css">    <link href="http://pingendo.github.io/pingendo-bootstrap/themes/default/bootstrap.css"    rel="stylesheet" type="text/css">'

    titlerow='<div class="row">          <div class="col-md-12">            <h1 class="text-center">'+headline+'</h1>          </div>        </div>'
    _HEAD='<head>'+_head+_scripts+'</head>'
    _BODY='<body>'+'<div class="section"><div class="container">'+titlerow+_body+'</div></div>'+'</body>'
    _HTML='<html>'+_HEAD+_BODY+'</html>'
    return _HTML


# In[18]:


def make_html5_plots(width=400,High='Open',Low='Close',currency='ETH',fiat='EUR',mins_points=[[5,4],[15,4],[15,16]]):
    result=[]
    output_name=currency+fiat
    for i in range( len(mins_points)):
        [mins,points]=mins_points[i]
        candel_stikcs=get_symbol_gdx(currency,minutes=mins,fiat=fiat,points=points)
        #_F15=fit_periods(candel_stikcs)
        _column_tbtransposed=make_column_plot(candel_stikcs,mins,points,width)
        #print(len(_column_tbtransposed))
        result.append(_column_tbtransposed)
    #    
    _scripts=scrips_from_components(result)
    _body=body_from_components(result)

            
    _HTML = HTMLfromScriptsBody(_scripts,_body,output_name)

    
    text_file = open(output_name+".html", "w")
    text_file.write(_HTML)
    text_file.close()
    
    publish_ftp_file(output_name+".html")
    
    output_file(output_name+".htm", title="GDAX monitor")
    save(row(column(result[0][0][0],result[0][1][0],result[0][2][0]),column(result[1][0][0],result[1][1][0],result[1][2][0]),column(result[2][0][0],result[2][1][0],result[2][2][0])))
    publish_ftp_file(output_name+".htm")


# In[19]:


def make_time_of_plot(ss):
    return ((str(ss)).split(' '))[1]


# In[20]:


def make_candle_plot_volume(df,title='Volume',plot_width=1000,normalized=False,one=-1,High='High',Low='Low',aspect=3,html5_components=False):
    #print(df)
    
    Volume='Volume'
    if normalized:
        norm=df['Volume'].iloc[one]
        df['nVolume']=df['Volume'].astype('float').divide(float(norm))
        #
        Volume='nVolume'

    w = 500*500*candel_frequency(df)/300
    #print('frequency:',candel_frequency(df)/60,' mins') 
    TOOLS = "pan,wheel_zoom,box_zoom,reset,save"
    start_time=make_time_of_plot(df['Date'].loc[0]) 
    cumtitle=title+" "+start_time

    _LinearFitFunction, ConstCoeff, LinearCoeff =fit_periods(df,High=Volume,Low=Volume,units='Vol')
    points=len(df.Date)
    mins=candel_frequency(df)/60
    X=np.arange(0,mins*points,mins)
    cumtitle=cumtitle+" [ {:.2g}".format(LinearCoeff*100)+" ]%/h"

    p = figure(x_axis_type="datetime", tools=TOOLS, plot_height=int(plot_width/aspect),plot_width=plot_width, title = cumtitle,x_axis_label='Time',y_axis_label='Volume')
    p.xaxis.axis_label_text_font_size = "8pt"
    p.yaxis.axis_label_text_font_size = "8pt"
    p.title.text_font_size = '8pt'
    p.xaxis.major_label_orientation = pi/4
    p.grid.grid_line_alpha=0.3
    p.line(df.Date.iloc[::-1],_LinearFitFunction(X), line_width=2)
    p.circle(df.Date.iloc[::-1],df[Volume].iloc[::-1], size=8)#fill_color="blue"
    
    if not html5_components:
        return p
    if html5_components:
        scr, div = components(p)
        return p, div, scr


# In[21]:


def make_column_plot(candel_stikcs,mins,points,width):
    pV1, divV1, scrV1 = make_candle_plot_volume(candel_stikcs,title=title_maker(mins,points),plot_width=width,html5_components=True)
    p1, div1, scr1 = make_candle_plot(candel_stikcs,title=title_maker(mins,points),plot_width=width,html5_components=True)
    p2, div2, scr2 = make_candle_plot(candel_stikcs,title=title_maker(mins,points),plot_width=width,normalized=True,html5_components=True)
    return [[p1, div1, scr1],[pV1, divV1, scrV1], [p2, div2, scr2]]


# In[22]:


def title_maker(mins=0,points=0):
    return "{:.2g}".format(points*mins/60)+"-hours plot, nu="+str(mins)+'min'


# In[23]:


def html5_column(_div):
    return '<div class="col-md-4">'+_div+'</div>'

def html5_row(_div):
    return '<div class="row">'+_div+'</div>'

