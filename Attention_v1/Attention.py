import utils

class Attention:
    def __init__(self,name):
        self.name = name
        self.sleep_score = 100
        self.cheat_score = 100
        self.stepout_score = 100
        self.Attenion_score = round((self.sleep_score + self.cheat_score + self.stepout_score)/3,2)
        self.bgcolor = None
        
        
    # 학생의 태도 점수를 reset 할때 사용가능
    def set_sleep_score(self, score):
        if 0 <= score <= 100:
            self.sleep_score = score
        print("0 에서 100 사이의 값으로 설정해주세요")
    
    def set_cheat_score(self, score):
        if 0 <= score <= 100:
            self.cheat_score = score
        print("0 에서 100 사이의 값으로 설정해주세요")
        
    def set_stepout_score(self, score):
        if 0 <= score <= 100:
            self.stepout_score = score
        print("0 에서 100 사이의 값으로 설정해주세요")
    
    # 이 학생에 대한 점수를 깎을 때 사용함
    def minus_sleep(self, minus):
        self.sleep_score -= minus
    
    def minus_cheat(self,minus):
        self.cheat_score -= minus
    
    def minus_stepout(self,minus):
        self.stepout_score -= minus
    
    # 현재 점수 확인용
    def get_each_score(self):
        return self.sleep_score, self.cheat_score, self.stepout_score
    
    def get_Attention_score(self):
        return self.Attention_score

    # 점수에 따른 배경 색 변경
    def set_bgcolor(self):
        if 0<= self.Attention_score <30:
            self.bgcolor = utils.RED
        elif 30<=self.Attenion_score < 70:
            self.bgcolor = utils.YELLOW
        elif 70<=self.Attenion_score < 100:
            self.bgcolor = utils.GREEN