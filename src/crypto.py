#!/usr/bin/env python3
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from statistics import mean 
import fire, json, os, string, sys, threading, random, model, sample, encoder, logging, time
from exchanges.bitfinex import Bitfinex
from twitter_scraper import get_tweets
import numpy as np
import tensorflow as tf

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# Console output debug prints
debug = True
# Timer trade threshold. Bot isn't the most secure btw.
sleeptime = 3600
# Model logic (trained to usually) 0.7-0.83 work well.
top_p = 0.7
# Temperature (refer to gpt-2 documentation)
temperature = 1
# Multuplier/Divider for top_p/length calc.(The more words the more token learning when positive, top_p decreases.)
# Adjust in small increments. The target is between 0.6 and 1 for top_p.
mx = 1
# Virtual USD
USD = 100

# End settings
mode = False
user = ""
running = False
BUY = False
dec = float(USD)
money = float(USD)
def scrape():
    arr = []
    for tweet in get_tweets('Bitcoin', pages=1):
        a = str(tweet['time'])
        b = str(tweet['text'])
        if "http" not in b:
            if ".com" not in b:
                arr.append([a, b])
    sort = sorted(arr,key=lambda x:time.strptime(x[0],"%Y-%m-%d %H:%M:%S"))
    out = str(sort).strip('[]')
    out1 = out.replace("',", ":")
    out2 = out1.replace("']", "")
    out3 = out2.replace("['", "")
    out4 = out3.replace("'", "")
    out5 = out4.replace('\\n', '')
    out6 = out5.replace('[', '')
    out7 = out6.replace(']', '')
    if debug == True:
        print(out7)
    return out7    

def help(bot, update):
    """Send a message when the command /help is issued."""
    update.message.reply_text('Initiate a /start command for crypto trade on demand. /timer to 6-hour timer trade.')

def timer(bot, update):
    comput = threading.Thread(target=interact_timer, args=(bot, update, top_p, mx, temperature,))
    comput.start()
    
def trade(bot, update):
    comput = threading.Thread(target=interact_model, args=(bot, update, top_p, mx, temperature,))
    comput.start()

def interact_model(bot, update, top_p, mx, temperature):
    global BUY
    global bitold
    global decold
    global mode
    model_name = '1558M'
    nsamples = 1
    batch_size = 1
    top_k = 0
    models_dir = 'models'
    tex = scrape()
    raw_text = str(tex)
#############################################
    if mode == False:
        cat = len(raw_text.split(" "))
        length = cat
        if length > 300:
            while length > 300:
                raw_text = raw_text.split(':', 1)[-1]
                length = len(raw_text.split(" "))
        cat = length
    tx = float(top_p)
    cax = float(cat)
    cay = float(mx)
    caz = float(cax * cay)
    ta = ((1-tx)/caz)
    top_p = ((tx) + (ta))
    if top_p > 1:
        top_p = 1
    if top_p < 0.005:
        top_p = 0.005
#############################################
    update.message.reply_text('Computing for 4 generations...')
    cache = []
    for x in range(0, 4):
        seed = random.randint(1431655765, 2863311530)
        models_dir = os.path.expanduser(os.path.expandvars(models_dir))
        if batch_size is None:
            batch_size = 1
        assert nsamples % batch_size == 0
        enc = encoder.get_encoder(model_name, models_dir)
        hparams = model.default_hparams()
        with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
            hparams.override_from_dict(json.load(f))
        if length is None:
            length = hparams.n_ctx // 2
        elif length > hparams.n_ctx:
            raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)
        with tf.Session(graph=tf.Graph()) as sess:
            context = tf.placeholder(tf.int32, [batch_size, None])
            np.random.seed(seed)
            tf.set_random_seed(seed)
            output = sample.sample_sequence(
                hparams=hparams, length=length,
                context=context,
                batch_size=batch_size,
                temperature=temperature, top_k=top_k, top_p=top_p
            )
            saver = tf.train.Saver()
            ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
            saver.restore(sess, ckpt)
            context_tokens = enc.encode(raw_text)
            generated = 0
            for _ in range(nsamples // batch_size):
                out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(batch_size)]
                })[:, len(context_tokens):]
                for i in range(batch_size):
                    generated += 1
                    text = enc.decode(out[i])
                    pika = text
                    stripes = pika.encode(encoding=sys.stdout.encoding,errors='ignore')
                    tigger = stripes.decode("utf-8")
                    meow= str(tigger)
                    analyzer = SentimentIntensityAnalyzer()
                    vs = analyzer.polarity_scores(meow)
                    data = vs.get('compound')
                    cache.append(data)
                    score = str(vs)
                    print("==========")
                    print("{:-<65} {}".format(meow, str(vs)))
                    print("==========")
                    lent = str(length)
                    print("Length: " + lent)
                    print("==========")
                    print("Output: " + meow)
                    print("==========")
                    tps = str(top_p)
                    print("top_p out: " + tps)
                    print("==========")
                    tpa = str(tx)
                    print("top_p in: " + tpa)
                    print("==========")
        sess.close()
    print(cache)
    sent = str(cache)
    rounded = mean(cache)
    rou = str(rounded)
    update.message.reply_text('Sentiment of generations are:' + rou)
    if rounded == 0:
        BUY = False
    if rounded > 0:
        BUY = True
    if rounded < 0:
        BUY = False
    if BUY == True:
        update.message.reply_text('Buy Signal...')
        print("BUY SIGNAL")
    if BUY == False:
        update.message.reply_text('Hold Signal...')
        print("HOLD SIGNAL")
        
def interact_timer(bot, update, top_p, mx, temperature):
    while True:
        global BUY
        global bitold
        global dec
        global mode
        global money
        if BUY == True:
            money = dec
            decimal = float(Bitfinex().get_current_price())
            one = decimal - bitold
            two = (one / decimal) # * 100
            mon = money * two
            money = mon + money
            bitold = decimal
            up = str(money)
            update.message.reply_text('Current money is: ' + up)
            print(money)
            print(dec)
            print(bitold)
            print("BUY SIGNAL CALC")
        if BUY == False:
            dec = money
            up = str(money)
            update.message.reply_text('Current money is: ' + up)
            print(money)
            print(dec)        
            print(bitold)
            print("HOLD SIGNAL CALC")
        model_name = '1558M'
        nsamples = 1
        batch_size = 1
        top_k = 0
        models_dir = 'models'
        tex = scrape()
        raw_text = str(tex)
#############################################
        if mode == False:
            cat = len(raw_text.split(" "))
            length = cat
            if length > 300:
                while length > 300:
                    raw_text = raw_text.split(':', 1)[-1]
                    length = len(raw_text.split(" "))
            cat = length
        tx = float(top_p)
        cax = float(cat)
        cay = float(mx)
        caz = float(cax * cay)
        ta = ((1-tx)/caz)
        top_p = ((tx) + (ta))
        if top_p > 1:
            top_p = 1
        if top_p < 0.005:
            top_p = 0.005
#############################################
        update.message.reply_text('Computing for 4 generations...')
        cache = []
        for x in range(0, 4):
            seed = random.randint(1431655765, 2863311530)
            models_dir = os.path.expanduser(os.path.expandvars(models_dir))
            if batch_size is None:
                batch_size = 1
            assert nsamples % batch_size == 0
            enc = encoder.get_encoder(model_name, models_dir)
            hparams = model.default_hparams()
            with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
                hparams.override_from_dict(json.load(f))
            if length is None:
                length = hparams.n_ctx // 2
            elif length > hparams.n_ctx:
                raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)
            with tf.Session(graph=tf.Graph()) as sess:
                context = tf.placeholder(tf.int32, [batch_size, None])
                np.random.seed(seed)
                tf.set_random_seed(seed)
                output = sample.sample_sequence(
                    hparams=hparams, length=length,
                    context=context,
                    batch_size=batch_size,
                    temperature=temperature, top_k=top_k, top_p=top_p
                )
                saver = tf.train.Saver()
                ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
                saver.restore(sess, ckpt)
                context_tokens = enc.encode(raw_text)
                generated = 0
                for _ in range(nsamples // batch_size):
                    out = sess.run(output, feed_dict={
                        context: [context_tokens for _ in range(batch_size)]
                    })[:, len(context_tokens):]
                    for i in range(batch_size):
                        generated += 1
                        text = enc.decode(out[i])
                        pika = text
                        stripes = pika.encode(encoding=sys.stdout.encoding,errors='ignore')
                        tigger = stripes.decode("utf-8")
                        meow= str(tigger)
                        analyzer = SentimentIntensityAnalyzer()
                        vs = analyzer.polarity_scores(meow)
                        data = vs.get('compound')
                        cache.append(data)
                        score = str(vs)
                        print("==========")
                        print("{:-<65} {}".format(meow, str(vs)))
                        print("==========")
                        lent = str(length)
                        print("Length: " + lent)
                        print("==========")
                        print("Output: " + meow)
                        print("==========")
                        tps = str(top_p)
                        print("top_p out: " + tps)
                        print("==========")
                        tpa = str(tx)
                        print("top_p in: " + tpa)
                        print("==========")
            sess.close()
        print(cache)
        sent = str(cache)
        rounded = mean(cache)
        rou = str(rounded)
        update.message.reply_text('Sentiment of generations are:' + rou)
        if rounded == 0:
            BUY = False
        if rounded > 0:
            BUY = True
        if rounded < 0:
            BUY = False
        if BUY == True:
            if b == False:
                bitold = float(Bitfinex().get_current_price())
            b = True
            update.message.reply_text('Buy Signal...')
            print(money)
            print(dec)
            print(bitold)
            print("BUY SIGNAL")
        if BUY == False:
            b = False
            update.message.reply_text('Hold Signal...')
            print(money)
            print(dec)
            print(bitold)
            print("HOLD SIGNAL")
        time.sleep(sleeptime)        

def error(bot, update):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update)

def main():
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    # Make sure to set use_context=True to use the new context based callbacks
    # Post version 12 this will no longer be necessary
    updater = Updater("BOTKEYBOTKEYBOTKEYBOTKEYBOTKEYBOTKEYBOTKEY", use_context=False)
    # Get the dispatcher to register handlers
    dp = updater.dispatcher
    # on different commands - answer in Telegram
    dp.add_handler(CommandHandler("timer", timer))
    dp.add_handler(CommandHandler("start", trade))
    dp.add_handler(CommandHandler("help", help))
    # on noncommand i.e message - echo the message on Telegram
    # log all errors
    dp.add_error_handler(error)
    # Start the Bot
    updater.start_polling()
    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()

if __name__ == '__main__':
    fire.Fire(main)
