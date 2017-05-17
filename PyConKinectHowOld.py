HEARTS_AND_MINDS_MODE = False
SHOW_PYTHON_VERSION = True
SHOW_AGE = True 
SHOW_GENDER = True
SHOW_ENGAGED = False
SHOW_IDENTITY = True

RATE_LIMIT_PER_MINUTE = 20


from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

import ctypes
import _ctypes
import pygame as pygame
import sys
import datetime
import timeit
import cognitive_face as CF
import requests

import config
#from .FaceFinder import face_finder_thread
import threading

if sys.hexversion >= 0x03000000:
    import _thread as thread
else:
    import thread


from threading import Thread
from queue import Queue, Empty
import pygame


key = config.COGNITIVE_FACES_KEY

surface_frame_queue = Queue(100)
faces_result_queue = Queue(10)

CF.Key.set(key)

def detect_faces(path):
    faces = CF.face.detect(path,landmarks=False, attributes="age,gender,smile,headPose,emotion")
    if faces:
        return faces
    return []

class BodyGameRuntime(object):
    def __init__(self):
        pygame.init()

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # Set the width and height of the screen [width, height]
        self._infoObject = pygame.display.Info()
        self._screen = pygame.display.set_mode((self._infoObject.current_w >> 1, self._infoObject.current_h >> 1), 
                                               pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)

        pygame.display.set_caption("Guess your Age!")

        # Loop until the user clicks the close button.
        self._done = False

        # Kinect runtime object, we want only color and body frames 
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body)

        # back buffer surface for getting Kinect color frames, 32bit color, width and height equal to the Kinect color frame size
        self._frame_surface = pygame.Surface((self._kinect.color_frame_desc.Width, self._kinect.color_frame_desc.Height), 0, 32)

        # here we will store skeleton data 
        self._bodies = None

        self._faces = []
        self._face_bodies = []

        self._update_oxford = 0
        self.python_logo_image = pygame.image.load('pylogo.png')

    
    """description of class"""
    def face_finder_thread(self):
        global surface_frame_queue
        global faces_result_queue
        while True:
            # get the surface frame queue, throw out all old stuff basically (we just want the top of the stack. we can pop off the back if we get full)
            frame = None
            bodies = None
            try:
                while True:
                    frame, bodies = surface_frame_queue.get(False)
            except Empty:            
                # our goal is to empty the queue so we have the latest
                pass
            except Exception as e:
                pass

            try:
                # now that we have a frame and bodies
                if frame is not None and bodies is not None:
                    # TODO: uncomment this. done for testing.
                    pygame.image.save(frame, 'file.jpg')
                    start = datetime.datetime.utcnow()
                    faces = detect_faces('file.jpg')

                    # wait time if we are trying to rate limit
                    if RATE_LIMIT_PER_MINUTE:
                        60 / RATE_LIMIT_PER_MINUTE
                        import time
                        time.sleep(60 / RATE_LIMIT_PER_MINUTE)

                    # Collect all face ids. This way we can batch the request.
                    faceIds = []
                    for face in faces:
                        faceId = face['faceId']
                        if faceId:
                            faceIds.append(faceId)
                    # if we have any face ids. We should try to get identities
                    identities = {}
                    if faceIds and SHOW_IDENTITY:
                        # get all of the person groups
                        response = requests.get("https://api.projectoxford.ai/face/v1.0/persongroups", 
                                                headers = {"Ocp-Apim-Subscription-Key":key})
                        person_groups = [i['personGroupId'] for i in response.json()]
                        # for each person group, try to identify this face.
                        for personGroup in person_groups:
                            identifyResults = CF.face.identify(faceIds, personGroup)
                            for result in identifyResults:
                                for candidate in result['candidates']:
                                    confidence = candidate['confidence']
                                    personData = CF.person.get(personGroup, candidate['personId'])
                                    name = personData['name']
                                    print('identified {0} with {1}% confidence'.format(name, str(float(confidence) * 100)))
                                    
                                    if result['faceId'] in identities:
                                        # if the new thing is more confident repplace
                                        _,oldConfidence = identities[result['faceId']]
                                        if oldConfidence < confidence:
                                            identities[result['faceId']] = personData, confidence
                                    else:
                                        identities[result['faceId']] = personData, confidence
                        # for each face, see if we now have the identity and add it to the object
                        faces_with_ids = []
                        for face in faces:
                            if face['faceId'] in identities:
                                personData,_ = identities[face['faceId']]
                                face['personData'] = personData
                            faces_with_ids.append(face)
                        faces = faces_with_ids

                    end = datetime.datetime.utcnow()
                    time_exec = end - start
                    print('Time: {}, Execution Time: {}, Result:{}'.format(end.strftime('%H:%M:%S'), time_exec, self._faces))
                    faces_result_queue.put((faces, bodies))
            except Exception as e:
                print(e)

    def draw_color_frame(self, frame, target_surface):
        target_surface.lock()
        address = self._kinect.surface_as_array(target_surface.get_buffer())
        ctypes.memmove(address, frame.ctypes.data, frame.size)
        del address
        target_surface.unlock()

    def run(self):
        # -------- Start Discovery Thread for Oxford --------
        t = Thread(target=self.face_finder_thread)
        t.start()

        global surface_frame_queue
        global faces_result_queue

        self.add_frame_to_queue = 0

        # -------- Main Program Loop -----------
        while not self._done:
             # Print framerate and playtime in titlebar.
            text = "FPS: {0:.2f}".format(self._clock.get_fps())
            pygame.display.set_caption(text)
            # --- Main event loop
            for event in pygame.event.get(): # User did something
                if event.type == pygame.QUIT: # If user clicked close
                    self._done = True # Flag that we are done so we exit this loop

                elif event.type == pygame.VIDEORESIZE: # window resized
                    self._screen = pygame.display.set_mode(event.dict['size'], 
                                               pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)
                    
            # --- Getting frames and drawing  
            # --- Woohoo! We've got a color frame! Let's fill out back buffer surface with frame's data 
            if self._kinect.has_new_color_frame():
                frame = self._kinect.get_last_color_frame()
                self.add_frame_to_queue+=1
                self.draw_color_frame(frame, self._frame_surface)
                frame = None

                if not self.add_frame_to_queue % 30:
                    surface_frame_queue.put((self._frame_surface, self._bodies))
            
            # Draw Chest Logos Using Kinect Data
            self.draw_logos_on_chests()

            #Draw Age Labels on Heads using Project Oxford returned Data
            self.draw_oxford_labels_on_surface()
    
            # Update stored body frames if we have a new one
            if self._kinect.has_new_body_frame():
                self._bodies = self._kinect.get_last_body_frame()

            # TODO: record locations of heads and their IDs. This will allow
            # us to track them through the frame.
            # the faces list then can relate to the kinect
            
            # --- copy back buffer surface pixels to the screen, resize it if needed and keep aspect ratio
            # --- (screen size may be different from Kinect's color frame size) 
            h_to_w = float(self._frame_surface.get_height()) / self._frame_surface.get_width()
            target_height = int(h_to_w * self._screen.get_width())
            surface_to_draw = pygame.transform.scale(self._frame_surface, (self._screen.get_width(), target_height));
            self._screen.blit(surface_to_draw, (0,0))
            surface_to_draw = None

            # Draw the curtain after, layering over the other surface
            self.draw_curtain()

            pygame.display.update()

            # --- Go ahead and update the screen with what we've drawn.
            pygame.display.flip()

            # --- Limit to 60 frames per second
            self._clock.tick(60)

        # Close our Kinect sensor, close the window and quit.
        self._kinect.close()
        pygame.quit()

    def draw_curtain(self):
        width = self._screen.get_width()
        height = self._screen.get_height()

        curtain_primary_color = pygame.color.Color('darkred')
        curtain_secondary_color = pygame.color.Color('gold')
        curtain_width = 40

        # NOTE: RECT ORDER: min x, min y, max x, max y

        # draw top rectangle and fill        
        rect = pygame.draw.rect(self._screen, curtain_primary_color, [0, 0, width, curtain_width], 2) 
        self._screen.fill(curtain_primary_color, rect)

        # draw right rectangle and fill
        right_rect = pygame.draw.rect(self._screen, curtain_primary_color, [width-curtain_width, 0, width, height], 2) 
        self._screen.fill(curtain_primary_color, right_rect)

        # draw left rectangle and fill
        left_rect = pygame.draw.rect(self._screen, curtain_primary_color, [0, 0, curtain_width, height], 2) 
        self._screen.fill(curtain_primary_color, left_rect)

        # Draw gold trace around inside of curtain
        pygame.draw.lines(self._screen, curtain_secondary_color, False,
                          [[curtain_width, height],[curtain_width,curtain_width], [width-curtain_width, curtain_width], [width-curtain_width, height]], 
                          2)

        # Draw GitHub Repo Address in middle top of screen (on curtain)
        font = pygame.font.SysFont("comicsansms", 30)
        text = font.render("https://github.com/crwilcox/KinectHowOld", True, pygame.color.THECOLORS['gold'])
        text_rect = text.get_rect(center=((self._screen.get_width() / 2), 17))
        self._screen.blit(text, text_rect)


    def get_body_head_position(self, body):
        head_joint = body.joints[JointType_Head]
        position = self._kinect.body_joint_to_color_space(head_joint)
        return position

    def get_body_chest_position(self, body):
        joint = body.joints[JointType_SpineMid]
        position = self._kinect.body_joint_to_color_space(joint)
        return position

    def draw_logos_on_chests(self):
        try:

            def scale_image(image, height):
                scale_percentage = height / float(image.get_height())
                scaled_height = int(image.get_height() * scale_percentage)
                scaled_width = int(image.get_width() * scale_percentage)
                return pygame.transform.scale(image, (scaled_width, scaled_height))

            if self._bodies:
                tracked_bodies =  (body for body in self._bodies.bodies if body.is_tracked and 
                                      body.joints[PyKinectV2.JointType_SpineMid].TrackingState is not PyKinectV2.TrackingState_NotTracked and
                                      body.joints[PyKinectV2.JointType_SpineShoulder].TrackingState is not PyKinectV2.TrackingState_NotTracked )

                # draw logos on tracked chests
                for i in tracked_bodies:
                    # distance between shoulders, distance between high and low spine
                    chest_position = self.get_body_chest_position(i)
                    shoulder_position = self._kinect.body_joint_to_color_space(i.joints[JointType_SpineShoulder])

                    height = chest_position.y - shoulder_position.y
                    height *= 1.25 # this distance is a hair short so blow it up a bit


                    # TODO: USE MOOD OF SUBJECT TO CHOOSE LOGO
                    # Expression API Doesn't appear done
                    scaled_image = scale_image(self.python_logo_image, height)
                    self._frame_surface.blit(scaled_image, (chest_position.x - (scaled_image.get_width()/2), chest_position.y - (scaled_image.get_height()/2)))
        except Exception as e:
            print("Exception in drawing logos on chest:", e)
   
    def get_python_version(self, age):
        release_years = [
            (0, "Assembly"),
            (1956, "FORTRAN"),
            # 1957
            (1958, "ALGOL II"),
            # 1959
            (1960, "ALGOL 60"),
            # 1961
            (1962, "FORTRAN IV"),
            # 1963 - 1965
            (1966, "FORTRAN 66"),
            # 1967
            (1968, "ALGOL 68"),
            (1970, "Pascal"),
            # 1971
            (1972, "Smalltalk"),
            # 1973 - 1976
            (1977, "FORTRAN 77"),
            (1978, "K&R C"),
            # 1979
            (1980, "Smalltalk"),
            # 1981 - 1987
            (1988, "Module-3"),
            (1989, "C89"),
            (1990, "Haskell"),
            (1991, "Python 0.9"),
            # 1992, 1993
            (1994, "Python 1.0"),
            # 1995, 1996
            (1997, "Python 1.5"),
            # 1998, 1999
            (2000, "Python 2.0"),
            (2001, "Python 2.2"),
            # 2002
            (2003, "Python 2.3"),
            (2004, "Python 2.4"),
            # 2005
            (2006, "Python 2.5"),
            #2007
            (2008, "Python 3.0"),
            (2009, "Python 3.1"),
            # 2010
            (2011, "Python 3.2"),
            (2012, "Python 3.3"),
            # 2013
            (2014, "Python 3.4"),
            (2015, "Python 3.5"),
            (2016, "Python 3.6"),
        ]

        year = datetime.datetime.now().year - age
        for release, name in reversed(release_years):
            if release < year:
                return name

        
    def user_engaged(self, face):
        threshold = 20
        if face and face['faceAttributes'] and face['faceAttributes']['headPose']:
            head_pose = face['faceAttributes']['headPose']
            roll = head_pose['roll']
            yaw = head_pose['yaw']
            
            # the closer the user is looking straight on the closer to 0 yaw and roll are.
            engagement_number = max(abs(roll), abs(yaw))
            if engagement_number <= threshold:
                return True
            else:
                return False
        return "CANNOT DETECT"

    def draw_oxford_labels_on_surface(self):
        try:
            if HEARTS_AND_MINDS_MODE:
                font = pygame.font.SysFont("comicsansms", 48)
                text = font.render("Winning Hearts and Minds mode enabled", True, pygame.color.THECOLORS['black'])
                self._frame_surface.blit(text, (900, 150))

            # check if we have faces to update from the background thread queue
            try:
                faces, bodies = faces_result_queue.get(False)
                if faces:
                    self._faces = faces
                    self._face_bodies = bodies
            except:
                pass

            #_faces, _face_bodies  use together
            # move labels based on _bodies as comparison. 
            # we should have tracked user. the label goes above their head
            tracked_head_points = []
            if self._face_bodies:
                tracked_bodies = [i for i in self._face_bodies.bodies if i.is_tracked and 
                                  i.joints[PyKinectV2.JointType_Head].TrackingState is not PyKinectV2.TrackingState_NotTracked]
            
                if tracked_bodies:                
                    #print("Kinect Head Position: x:{} y:{}".format(position.x, position.y))
                    tracked_head_points = [(i, self.get_body_head_position(i)) for i in tracked_bodies]

            def is_point_contained(point, top_y, bottom_y, left_x, right_x, pixel_variation=50):
                '''
                is point contained. Assumes top left 0,0
                '''
                if(point.x < right_x + pixel_variation and 
                    point.x > left_x - pixel_variation and 
                    point.y < bottom_y + pixel_variation and
                    point.y > top_y - pixel_variation):
                    return True
                else:
                    return False

            # track ids of bodies. This way we can record old info. this 
            # makes it seem faster. also, if we have outages of wifi old returning people will get bubbles
            for face in self._faces:
                top_y = face['faceRectangle']['top']
                left = face['faceRectangle']['left']
                height = face['faceRectangle']['height']
                width = face['faceRectangle']['width']
                #print("Oxford Face Position: Top:{} Left:{} Width:{} Height:{}".format(top_y, left, width, height))
               
                # only put the info on if this head is tracked.
                if tracked_head_points: 
                    
                    # draw age data for tracked heads
                    bodies = [i for i in tracked_head_points if is_point_contained(i[1], top_y, top_y+height, left, left + width)]
                    if bodies:
                        # TODO: take the first for now. in the future we should find the closest match to the head marker
                        body, colorspace_point = bodies[0]
                        try:
                            this_body = next((i for i in self._bodies.bodies if i.tracking_id == body.tracking_id))
                            head_position = self.get_body_head_position(this_body)
                        
                            # Draw the Age Above the face
                            font = pygame.font.SysFont("comicsansms", 48)
                            age = face['faceAttributes']['age']
                            
                            strings_to_draw = []

                            # Based on options configured, display different things.
                            if HEARTS_AND_MINDS_MODE:
                                age = int(age * .65)

                            if SHOW_IDENTITY:
                                strings_to_draw.append(face['personData']['name'])

                            if SHOW_AGE:
                                strings_to_draw.append(age)
                                
                            if SHOW_GENDER:
                                strings_to_draw.append(face['faceAttributes']['gender'])
                            
                            if SHOW_ENGAGED:
                                strings_to_draw.append(f"engaged: {str(self.user_engaged(face))} (kinect: {str(this_body.engaged)})")
                           
                            if SHOW_PYTHON_VERSION:
                                strings_to_draw.append(f"python version: {self.get_python_version(age)}")

                            # TODO: if we have the identity, add it.
                            height = (len(strings_to_draw) * 60) + 50;
                            line_height = 60;
                            for string in strings_to_draw:
                                text = font.render(str(string), True, pygame.color.THECOLORS['black'], pygame.color.THECOLORS['white'])
                                self._frame_surface.blit(text, (head_position.x + 75, max(head_position.y - height, 0)))
                                height = height - line_height
                                


                        except StopIteration:
                            pass # this is fine. we just didn't find the body
        except Exception as e:
            print("Exception in drawing text over head:", e)


#face_finder_thread().run()
if __name__ == "__main__":
    __main__ = "Guess Your Age Game"
    game = BodyGameRuntime();
    game.run();
