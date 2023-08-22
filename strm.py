import streamlit as st 
import biosppy
import time 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import pyhrv.tools as tools
import pyhrv.time_domain as td
import pyhrv.frequency_domain as fd
import pyhrv.nonlinear as nl
import pyhrv
import pandas as pd
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as pt
from matplotlib import rcParams
import seaborn as sns
import serial
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score
import sklearn.metrics as met
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import glob

def serial_ports():
    """ Lists serial port names

        :raises EnvironmentError:
            On unsupported or unknown platforms
        :returns:
            A list of the serial ports available on the system
    """
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this excludes your current terminal "/dev/tty"
        ports = glob.glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.*')
    else:
        raise EnvironmentError('Unsupported platform')

    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result

co = serial_ports()

st.title("Sleep Apnea Detection")
add_selectbox = st.sidebar.selectbox(
    "Menu",("Home", "Data", "HRV","Hasil")
)


st.sidebar.write("Deteksi Langsung")
if st.sidebar.button("Deteksi"):
    fig = plt.figure(figsize=(7,3))
    ax = fig.add_subplot()
    ser = serial.Serial(co[0],115200)
    ser.close()
    x = []
    xx= []
    df =[]
    i=0
    l = 37500
    ser.open()
    for i in range(l):
        try:
            ser1 = ser.readline().decode("ascii")
            ser2 = float(ser1)
            x.append(ser2)
            print(ser2)
        except :
            pass
    ser.close()
    df = pd.DataFrame(x)
    signal = np.array(df[df.columns[0]])
    st.write(fig)
    t, filtered_signal, rpeaks = biosppy.signals.ecg.ecg(signal, sampling_rate=100,show=False)[:3]
    hr = biosppy.signals.tools.get_heart_rate(beats=rpeaks,sampling_rate=100,smooth=True)
    hr = hr["heart_rate"]
    ept =tools.plot_ecg(signal, sampling_rate = 100, interval=[0, 20])
    st.write(ept["ecg_plot"])
    st.header("Time Domain")
    ax.grid()
    ax.set_title("Heart Rate Variability")
    ax.set_xlabel("Sequence (n)")
    ax.set_ylabel("RR Interval (ms)")
    ax.plot(hr)
    st.write(fig)
    col1,col2 = st.columns(2)
    with col2 :
        results = td.sdnn(rpeaks=t[rpeaks])
        sdnn = results['sdnn']
        st.subheader("SDNN")
        st.write(str(sdnn))
        results = td.rmssd(rpeaks=t[rpeaks])
        rmssd = results['rmssd']
        st.subheader("RMSSD")
        st.write(str(rmssd))
    with col1 :
        results = td.nn50(rpeaks=t[rpeaks])
        pnn50 = results['pnn50']
        st.subheader("PNN50")
        st.write(str(pnn50))
        results = td.sdsd(rpeaks=t[rpeaks])
        sdsd = results["sdsd"]
        st.subheader("SDSD")
        st.write(str(sdsd))
    result = fd.welch_psd(rpeaks=t[rpeaks])
    st.header("Frequency Domain")
    st.write(result["fft_plot"])
    result = fd.welch_psd(rpeaks=t[rpeaks],show=False)
    LFHF_Ratio = result["fft_ratio"]
    st.subheader("LF/HF Ratio")
    st.write(str(LFHF_Ratio))
    results = pyhrv.nonlinear.poincare(rpeaks=t[rpeaks])
    st.header("Nonlinear - Poincare")
    st.write(results["poincare_plot"])
    results = pyhrv.nonlinear.poincare(rpeaks=t[rpeaks],show=False)
    col1,col2 = st.columns(2)
    with col1 :
        sd1 = results['sd1']
        st.subheader("SD 1")
        st.write(str(sd1))
    with col2 :
        st.subheader("SD 2")
        sd2 = results['sd2']
        st.write(str(sd2))
    df_sd2 = pd.DataFrame({"SD2" : [sd2]})
    df_sdnn = pd.DataFrame({"SDNN":[sdnn]})
    df_lfhf = pd.DataFrame({"LF/HF":[LFHF_Ratio]})
    df_pnn50 = pd.DataFrame({"pNN50":[pnn50]})
    df_rmssd =pd.DataFrame({"RMSSD":[rmssd]})
    df_sd1 = pd.DataFrame({"SD1":[sd1]})
    df_sdsd =pd.DataFrame({"SDSD":[sdsd]})
    df_result = pd.concat([df_sd2,
                                df_sdnn,
                                df_lfhf,
                                df_pnn50,
                                df_rmssd,
                                df_sd1,
                                df_sdsd,],axis=1)

    df1=pd.DataFrame(df_result)
    data=pd.read_excel('9.xlsx')
    Y = data['0/1']
    X = data.drop(['0/1'], axis = 1)
    x_training, x_testing, y_training, y_testing = train_test_split(X, Y, test_size = 0.33, random_state = 0)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_training,y_training)
    prediction = knn.predict(x_testing)
    prediction = knn.predict(df1)
    if prediction[0] == 1 :
        st.header("Apnea")
    else : 
        st.header("Normal")


    

else :
    fig = plt.figure(figsize=(7,3))
    ax = fig.add_subplot()

        
    if add_selectbox == "Home":
        col1,col2 = st.columns(2)
        with col1 :
            st.image("its.png",width=150,clamp=True)
        with col2:
            st.image("gem.png",use_column_width="auto")
        st.header("Piranti Cerdas, Sistem Benam & IoT")
        st.header("Slumber Squad")
        st.subheader("Anggota Kelompok :")
        col1,col2 = st.columns(2)
        with col1 : 
            st.write("Firdausa Sonna Anggara Resta")
            st.write("Rima Amalia")
            st.write("Mu'afa Ali Syakir")
        with col2 :
            st.write("07311940000010")
            st.write("5023201005")
            st.write("5023211023")
        

    elif add_selectbox =="Data":
        ser = serial.Serial(co[0],115200)
        ser.close()
        x = []
        xx= []
        df =[]
        i=0
        l = 37500
        st.write("ini untuk pengambilan data")
        ser.open()
        for i in range(l):
            try:
                ser1 = ser.readline().decode("ascii")
                ser2 = float(ser1)
                x.append(ser2)
                print(ser2)
            except :
                pass
        ser.close()
        df = pd.DataFrame(x)
        signal = np.array(df[df.columns[0]])
        ax.plot(x)
        st.write(fig)
        def convert_df(df):
            return df.to_csv(index=False,sep=" ").encode('utf-8')
        try:
            csv = convert_df(df)
            st.download_button(label="Download data as csv",data=csv,file_name='Data Sample.csv', mime='text/csv',)    
        except:
            pass
        ser.__del__()

        


    elif add_selectbox == "HRV":
        st.write("Memploting HRV dan Fitur lainnya")
        uploaded_file = st.file_uploader("Choose a ECG file")
        st.header("ECG plot")
        if uploaded_file is not None:
            dk = pd.read_csv(uploaded_file,sep='\s+',header=[0])
            dk = pd.DataFrame(dk)
            signal = np.array(dk[dk.columns[0]])
            t, filtered_signal, rpeaks = biosppy.signals.ecg.ecg(signal, sampling_rate=100,show=False)[:3]
            hr = biosppy.signals.tools.get_heart_rate(beats=rpeaks,sampling_rate=100,smooth=True)
            hr = hr["heart_rate"]
            ept =tools.plot_ecg(signal, sampling_rate = 100, interval=[0, 20])
            st.write(ept["ecg_plot"])

        with st.sidebar:
            st.write("HRV Feature")
        if st.sidebar.button("Generate"):
            st.header("Time Domain")
            ax.grid()
            ax.set_title("Heart Rate Variability")
            ax.set_xlabel("Sequence (n)")
            ax.set_ylabel("RR Interval (ms)")
            ax.plot(hr)
            st.write(fig)
            col1,col2 = st.columns(2)
            with col2 :
                results = td.sdnn(rpeaks=t[rpeaks])
                sdnn = results['sdnn']
                st.subheader("SDNN")
                st.write(str(sdnn))
                results = td.rmssd(rpeaks=t[rpeaks])
                rmssd = results['rmssd']
                st.subheader("RMSSD")
                st.write(str(rmssd))
            with col1 :
                results = td.nn50(rpeaks=t[rpeaks])
                pnn50 = results['pnn50']
                st.subheader("PNN50")
                st.write(str(pnn50))
                results = td.sdsd(rpeaks=t[rpeaks])
                sdsd = results["sdsd"]
                st.subheader("SDSD")
                st.write(str(sdsd))
            result = fd.welch_psd(rpeaks=t[rpeaks])
            st.header("Frequency Domain")
            st.write(result["fft_plot"])
            result = fd.welch_psd(rpeaks=t[rpeaks],show=False)
            LFHF_Ratio = result["fft_ratio"]
            st.subheader("LF/HF Ratio")
            st.write(str(LFHF_Ratio))
            results = pyhrv.nonlinear.poincare(rpeaks=t[rpeaks])
            st.header("Nonlinear - Poincare")
            st.write(results["poincare_plot"])
            results = pyhrv.nonlinear.poincare(rpeaks=t[rpeaks],show=False)
            col1,col2 = st.columns(2)
            with col1 :
                sd1 = results['sd1']
                st.subheader("SD 1")
                st.write(str(sd1))
            with col2 :
                st.subheader("SD 2")
                sd2 = results['sd2']
                st.write(str(sd2))
            df_sd2 = pd.DataFrame({"SD2" : [sd2]})
            df_sdnn = pd.DataFrame({"SDNN":[sdnn]})
            df_lfhf = pd.DataFrame({"LF/HF":[LFHF_Ratio]})
            df_pnn50 = pd.DataFrame({"pNN50":[pnn50]})
            df_rmssd =pd.DataFrame({"RMSSD":[rmssd]})
            df_sd1 = pd.DataFrame({"SD1":[sd1]})
            df_sdsd =pd.DataFrame({"SDSD":[sdsd]})
            df_result = pd.concat([df_sd2,
                                df_sdnn,
                                df_lfhf,
                                df_pnn50,
                                df_rmssd,
                                df_sd1,
                                df_sdsd,],axis=1)

            df1=pd.DataFrame(df_result)
            def convert_df(df1):
                return df1.to_csv(index=False).encode('utf-8')
            csv = convert_df(df1)
            st.download_button(label="Download data as csv",data=csv,file_name='Fitur.csv', mime='text/csv',)    

    elif add_selectbox == "Hasil":
        data=pd.read_excel('9.xlsx')
        Y = data['0/1']
        X = data.drop(['0/1'], axis = 1)
        x_training, x_testing, y_training, y_testing = train_test_split(X, Y, test_size = 0.33, random_state = 0)
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(x_training,y_training)
        prediction = knn.predict(x_testing)
        uploaded_file = st.file_uploader("Chose Feature File")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            prediction = knn.predict(df)
            if prediction[0] == 1 :
                st.header("Apnea")
            else : 
                st.header("Normal")












    
    
    



    
    

