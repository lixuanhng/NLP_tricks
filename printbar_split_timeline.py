"""
Sometimes we would like to check the epoch results when training a model,
it is quite necessary to print split_timeline in format.
"""
import tensorflow as tf

@tf.function
def printbar():
    # Provides the time since epoch in seconds.
    today_ts = tf.timestamp()%(24*60*60)
    
    # UTC timezone(PST+8) transformation
    hour = tf.cast(today_ts//3600+8,tf.int32)%tf.constant(24)  
    minute = tf.cast((today_ts%3600)//60,tf.int32)
    second = tf.cast(tf.floor(today_ts%60),tf.int32)  
    # tf.floor: replapce the num with '0' after decimal point
    
    def timeformat(m):
        # if m is tens digit，print it out directly；if ones digit, put on a '0' in front of m, then print it
        if tf.strings.length(tf.strings.format("{}",m))==1:
            return(tf.strings.format("0{}",m))
        else:
            return(tf.strings.format("{}",m))
    
    timestring = tf.strings.join([timeformat(hour),timeformat(minite),
                timeformat(second)],separator = ":")
    tf.print("=========="*8+timestring)
