import time
from datetime import datetime
from unicodedata import name

global score, running_times 

def Attention_score(eye, cheat, step_out):
    total_score = (eye + cheat + step_out) / 3
    return total_score

def Eye_score(eye_closed, start_time, end_time, sec = 1):
    score = 100
    blink_standard = 3
    blink_count = 0
    
    if eye_closed == True:
        pass
        
    elif blink_count >= blink_standard:
        pass
    
    return score

def Hpe_score(text, start_time, end_time, sec=1):
    # score의 경우에 글로벌 변수를 써야할 듯
    score = 100
    
    total_time = end_time - start_time
    running_times = 0
    cheat_time = 0
    dir_per_ms5 = 0
    
    dir_per_ms5 += time.time() - start_time
    cheat_time += time.time() - start_time
    time_log = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    
    cheat_count = 0
    cheat_standard = 3
    
    # 0.5초 마다 방향과 시간출력
    # 캄퓨타 시간 받아오기
    if (dir_per_ms5//(sec/2) >=1) & (text == 'cheat'):
        dir_per_ms5 = 0
        print(time_log, text, cheat_time)
        
        # 정해진 시간(=10초)이 되면 카운트 1 올리기
        # cheat_count 가 '+1' 되면 cheat_time 을 초기화
        if cheat_time//10 >= 1:
            cheat_count += 1
            running_times += cheat_time
            cheat_time = 0
            print(time_log, "cheat_count is:", cheat_count)
            print(running_times)
            
        # if (시간차가 1분 보다 작으면) & (cheat_count>3):
            if (running_times < sec*60) & (cheat_count>=cheat_standard):
                score -= 5
                cheat_count = 0
                # print("Your score is %d"%(score))
                # print(running_times)
                running_times = 0
        
    # 그러다가 정면을 보면 time과 cnt 는 0으로 초기화
    elif (text == 'Forward'):
        cheat_time = 0
        dir_per_ms5 = 0
        
    return score

def StepOut_Score(text, per_min = 1, sec = 0.5):
    score = 100
    if aaa:
        pass
    return score

