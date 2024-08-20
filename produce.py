# -*- coding: utf-8 -*-
from autoprocess import autorun,autorun2
from tools.i18n.i18n import I18nAuto, scan_language_list
i18n = I18nAuto()

train_file_name="shoulinrui.m4a"
train=False
gpu="1-2" #多个要-，单个就打数字
reference="resources/slice/shoulinrui.m4a/shoulinrui.m4a_0001110720_0001286400.wav"
text="这段语音是通过服务器上的自动生成函数获得的语音，使用了1和2两张显卡，全中文语言，效果不错"
text_language=i18n("中文")

prompt_text="个性化和移动端订单的增长，让北美门店叫苦不迭"
prompt_language=i18n("中文")

def auto_v1(train_file_name,train,gpu,reference,text,text_language,prompt_text,prompt_language):
    autorun(train_file_name,train,gpu,reference,text,text_language,prompt_text,prompt_language)
    autorun2(train_file_name,train,gpu,reference,text,text_language,prompt_text,prompt_language)

auto_v1(train_file_name,train,gpu,reference,text,text_language,prompt_text,prompt_language)