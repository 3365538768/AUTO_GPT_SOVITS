# -*- coding: utf-8 -*-
import os
from autoprocess import autorun,autorun2
from tools.i18n.i18n import I18nAuto, scan_language_list
i18n = I18nAuto()
import librosa
import shutil

def auto_v1(train_file_name,train,gpu,reference,text,text_language,prompt_text,prompt_language,gpt_path,sovits_path,show):
    autorun(train_file_name,train,gpu,reference,text,text_language,prompt_text,prompt_language,show)
    autorun2(train_file_name,train,gpu,reference,text,text_language,prompt_text,prompt_language,gpt_path,sovits_path,show)

def auto_v1_test():
    reference="resources/slice/shoulinrui.m4a/shoulinrui.m4a_0001110720_0001286400.wav"
    prompt_text="个性化和移动端订单的增长，让北美门店叫苦不迭"
    text="他深吸一口气，缓缓地握紧拳头，目光坚定地望向远方，仿佛看见了未来的光芒，我一定会赢，一定会的！"
    train = False
    gpu = "0"  # 多个要-，单个就打数字
    text_language = i18n("中文")
    prompt_language = i18n("中文")
    train_file_name = "shoulinrui.m4a"
    exp_name = "exp_" + train_file_name
    gpt_path = "GPT_weights_v2/" + exp_name + "-e15.ckpt"
    sovits_path = 'SoVITS_weights_v2/' + exp_name + '_e8_s192.pth'
    show = False
    auto_v1(train_file_name,train,gpu,reference,text,text_language,prompt_text,prompt_language,gpt_path,sovits_path,show)

def find_text(wavname,list_path):
    with open(list_path,"r",encoding="utf-8") as f:
        for line in f:
            if wavname in line:
                return line.split("|")[-1]

def slice_auto(texts,train,folder_path,train_file_name,gpu,text_language,prompt_language,gpt_path,sovits_path,show):
    if train:
        autorun(train_file_name, True, gpu, "", "", text_language, "", prompt_language, show)
    for filename in os.listdir(folder_path):
        file_path=os.path.join(folder_path,filename)
        if filename.endswith(".wav"):
            y,sr=librosa.load(file_path,sr=None)
            duration=librosa.get_duration(y,sr)
            if(duration>3.0 and duration<10.0):
                for text in texts:
                    print(f"{filename}——————3.0<{duration}<10.0——————满足时长")
                    prompt_text=find_text(filename,"resources/asr/shoulinrui.m4a/shoulinrui.m4a.list")
                    print(f"对应参考文本——————{text}")
                    auto_v1(train_file_name, False, gpu, file_path, text, text_language, prompt_text, prompt_language, gpt_path,
                            sovits_path,show)
                    print("输出成功")
            else:
                print(f"{filename}——————{duration}不满足")


train_file_name="shoulinrui.m4a"
exp_name = "exp_" + train_file_name
gpu="0" #多个要-，单个就打数字
text_language=i18n("中文")
prompt_language=i18n("中文")
gpt_path = "GPT_weights_v2/" + exp_name + "-e15.ckpt"
sovits_path = 'SoVITS_weights_v2/' + exp_name + '_e8_s192.pth'
show=True
train=True
# texts=["这段语音是通过服务器上的自动生成函数获得的语音，使用了1和2两张显卡，全中文语言，效果不错",
#        "“为什么！为什么总是这样！”他愤怒地拍打着桌子，眼神中闪烁着不甘的泪光，“我已经付出了这么多，为什么还是得不到回报？”声音在房间里回荡着，透着绝望和疲惫。",
#        "他沉默了片刻，颓然地跌坐在椅子上，双手无力地捂住脸庞，声音渐渐变得低沉，“难道，所有的努力都只是徒劳吗？我真的这么失败吗……”",
#        "他深吸一口气，缓缓地握紧拳头，目光坚定地望向远方，仿佛看见了未来的光芒，“我一定会赢，一定会的！”",
#        "各位观众朋友，大家晚上好。我们刚刚收到一条令人震惊的消息！今天下午三点，在市中心发生了一起严重的交通事故！”播音员的声音中透着紧张和急促，“一辆失控的卡车突然冲上了人行道，撞击了多名正在路边等待过马路的行人！现场一片混乱，令人触目惊心。",
#        "播音员稍稍停顿，继续报道，“据目击者称，事故发生时，卡车司机似乎失去了对车辆的控制。我们得知，伤者中包括几名年幼的孩子，他们的伤情目前非常严重。”",]
texts=["小宝贝，再不睡觉我就要亲你了哦"]
folder_path="resources/slice/shoulinrui.m4a/"  #参考音频的文件夹，会对其进行遍历，判断是否满足3~10s

slice_auto(texts,train,folder_path,train_file_name,gpu,text_language,prompt_language,gpt_path,sovits_path,show)
