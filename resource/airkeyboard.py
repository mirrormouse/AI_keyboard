import cv2
import torch
import math
import pyautogui
modelpath='./weights/best_ten2.pt'
model = torch.hub.load('../yolov5', 'custom', path=modelpath, source='local')
# VideoCapture オブジェクトを取得
capture = cv2.VideoCapture(0)

# 便利なクラスと関数
class rect:
  def __init__(self,l,t,r,b,c,n,n2=""):
      self.l=l
      self.t=t
      self.r=r
      self.b=b
      self.c=c
      self.n=n
      self.n2=n2
  def center(self):
    return point((self.l+self.r)//2, (self.t+self.b)//2)
  def contain(self,p):
    if self.l<p.x and p.x<self.r and self.t<p.y and p.y<self.b:
      return True
    else:
      return False

class point:
  def __init__(self,x,y):
      self.x=x
      self.y=y

def Length(p1,p2):
  d2=(p1.x-p2.x)**2+(p1.y-p2.y)**2
  return math.sqrt(d2)

def drawtext(img,text,p,color,font=1,thickness=2):
  return cv2.putText(img,
              text=text,
              org=(p.x,p.y),
              fontFace=cv2.FONT_HERSHEY_SIMPLEX,
              fontScale=font,
              color=color,
              thickness=thickness,
              lineType=cv2.LINE_4)


def drawrect(img,R,color,thickness):
  return cv2.rectangle(img, (R.l, R.t), (R.r, R.b), color, thickness=thickness)

# キーボード作成
keyboard=[
    ["1","2","3","4","5","6","7","8","9","0","-","^","backspace"],
    ["q","w","e","r","t","y","u","i","o","p","@","["],
    ["a","s","d","f","g","h","j","k","l",";",":","]"],
    ["z","x","c","v","b","n","m",",",".","/","\\"],
]
shiftkeyboard=[
    ["!",'"',"#","$","%","&","'","(",")","","=","~",""],
    ["","","","","","","","","","","`","{"],
    ["","","","","","","","","","+","*","}"],
    ["","","","","","","","","","","_"],
]
zerorect=rect(0,0,0,0,0,"zero")


def makeROIrect(p,s,c,c2):
    return rect(p.x,p.y,p.x+s.x,p.y+s.y,0,c,c2)

#キーボードを画面上に配置するために必要な位置指定
size=point(45,45)
uple=point(10,10)
step=point(12,30)
#キーボードを画面上に作成
def createROI():
    ALLROI=[]

    ALLROI.append(rect(uple.x+2*step.x+12*size.x,uple.y+size.y,uple.x+2*step.x+13*size.x,uple.y+3*size.y,0,"enter"))
    for i,(l,r) in enumerate(zip(keyboard,shiftkeyboard)):
        b=point(uple.x+i*step.x,uple.y+i*size.y)
        for j,(c,c2) in enumerate(zip(l,r)):
            p=point(b.x+size.x*j,b.y)
            ALLROI.append(makeROIrect(p,size,c,c2))
    ALLROI.append(rect(uple.x+3*step.x+3*size.x,uple.y+4*size.y,uple.x+3*step.x+7*size.x,uple.y+5*size.y,0," "))
    #ALLROI.append(rect(uple.x+3*step.x+7*size.x,uple.y+4*size.y,uple.x+3*step.x+8*size.x,uple.y+5*size.y,0,"Change"))
    return ALLROI


ROIs=createROI()
shiftROI=rect(uple.x+3*step.x+0*size.x,uple.y+4*size.y,uple.x+3*step.x+3*size.x,uple.y+5*size.y,0,"shift")



#YOLOで検出。OpenCVとYoloでRGBの順序が違うので注意。
def YoloDetection(img):
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  result=model(img_rgb)
  objects=result.pandas().xyxy[0]
  num=len(objects.xmin)
  Rects=[zerorect]
  for i in range(num):
    R=rect(int(objects.xmin[i]),int(objects.ymin[i]),int(objects.xmax[i]),int(objects.ymax[i]),objects.confidence[i],objects.name[i])
    Rects.append(R)
  return Rects

#シフトキーの実装
def drawshift(res,shiftstate,roi,arect):
    textpoint=point(roi.center().x-size.x//2,roi.center().y+size.y//3)
    res=drawtext(res,roi.n,textpoint,(0,0,0),0.5,1)
    res=drawrect(res,roi,(255,0,0),3)
    if arect.c>0:
        res=cv2.circle(res, (arect.center().x, arect.center().y), 5, (255, 255, 255), thickness=-1)
    rc=res.copy()
    if shiftstate:
        rc=drawrect(rc,roi,(255,0,0),-1)
        res = cv2.addWeighted(rc, 0.6, res, 0.4, 0)
    else:
        rc=drawrect(rc,roi,(255,0,0),-1)
        res = cv2.addWeighted(rc, 0.2, res, 0.8, 0)
    return res

#GUI全体の実装
def makebaseGUI(img,arect,nowstate,shiftrect,shiftstate):
    res=img.copy()
    for roi in ROIs:
        textpoint=point(roi.center().x-size.x//4,roi.center().y+size.y//3)
        shifttextpoint=point(roi.center().x-size.x//4,roi.center().y-size.y//8)
        if roi.n=="enter":
            enterpoint=point(roi.center().x-size.x//3,roi.center().y+size.y//4)
            res=drawtext(res,roi.n,enterpoint,(0,0,0),0.4,1)
        elif roi.n=="backspace":
            bspoint=point(roi.center().x-size.x//3,roi.center().y+size.y//4)
            res=drawtext(res,"Bs",bspoint,(0,0,0),0.5,1)
        else:
            res=drawtext(res,roi.n,textpoint,(0,0,0),0.5,1)
            res=drawtext(res,roi.n2,shifttextpoint,(0,0,0),0.5,1)
    newstate=-1
    for i,roi in enumerate(ROIs):
        #print(f"{roi.l},{roi.t},{roi.r},{roi.b}")
        res=drawrect(res,roi,(255,0,0),3)
        rc=res.copy()
        rc=drawrect(rc,roi,(255,0,0),-1)
        if (nowstate==i or arect.c>0.5) and roi.contain(arect.center()):
            res=cv2.circle(res, (arect.center().x, arect.center().y), 5, (255, 255, 255), thickness=-1)
            res = cv2.addWeighted(rc, 0.6, res, 0.4, 0)
            newstate=i
        else:
            res = cv2.addWeighted(rc, 0.2, res, 0.8, 0)
    res=drawshift(res,shiftstate,shiftROI,shiftrect)
    #print(f"now:{nowstate},new:{newstate}")
    if newstate>=0 and nowstate!=newstate:
        if shiftstate:
            pyautogui.press(ROIs[newstate].n2)
        else:
            pyautogui.press(ROIs[newstate].n)
    nowstate=newstate
    return res,nowstate

def Abstract(rects):
    arect=rects[0]
    for r in rects:
        if r.n=="fingers" and r.c>arect.c:
            arect=r
    return arect

def shiftjudge(rects,nowshift):
    newshift=False
    shiftrect=rects[0]
    for r in rects:
        if (nowshift or r.c>0.5) and shiftROI.contain(r.center()) and r.n=="fingers":
            newshift=True
            if r.c>shiftrect.c:
                shiftrect=r
    return newshift,shiftrect

def validityjudge(rects,arect):
    valid=True
    threshold=50
    c1=arect.center()
    for r in rects:
        if r.n=="afinger":
            c2=r.center()
            if Length(c1,c2)<threshold and arect.c<=r.c:
                valid=False
    return False

record=[]
nowstate=-1
while(True):
    nowshift=False
    ret, frame = capture.read()
    # resize the window
    Rects=YoloDetection(frame)   
    arect=Abstract(Rects)
    valid=validityjudge(Rects,arect)
    nowshift,shiftrect=shiftjudge(Rects,nowshift)
    resframe,nowstate=makebaseGUI(frame,arect,nowstate,shiftrect,nowshift)
    windowsize = (1920, 1080)
    resframe = cv2.resize(resframe, windowsize)
    cv2.imshow('title',resframe)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()