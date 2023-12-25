from utils3D import compile_shader, read_pointcloud, setup_glfw, flatten_pointcloud_with_color
import glfw
from OpenGL.GL import *
import pyrr
import numpy as np
from camera import Camera
from sklearn.neighbors import NearestNeighbors
import sounddevice as sd
import soundfile as sf
from gen_utils import load_models, generate_interp
import torch

#Callbacks for events:
MOVE_SPEED = 0.1
WIDTH = 1280
HEIGHT = 1280
lastX, lastY = WIDTH / 2, HEIGHT / 2
first_mouse = True
left, right, forward, backward = False, False, False, False
mouse_l_pressed = False
last_click_coords = None
last_last_click_coords = None
last_click_processed = True
last_selected_idx = -1
selected_idx = -1
models_data = load_models()
models_data['emae'].load_state_dict(models_data['emae_sd']['base'])
models_data['gpt'].load_state_dict(models_data['gpt_sd']['NSynth'])
models_data['gpt'] = models_data['gpt'].to(dtype=torch.float16)

# the keyboard input callback
def key_input_clb(window, key, scancode, action, mode):
    global left, right, forward, backward
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)

    if key == glfw.KEY_W and action == glfw.PRESS:
        forward = True
    elif key == glfw.KEY_W and action == glfw.RELEASE:
        forward = False
    if key == glfw.KEY_S and action == glfw.PRESS:
        backward = True
    elif key == glfw.KEY_S and action == glfw.RELEASE:
        backward = False
    if key == glfw.KEY_A and action == glfw.PRESS:
        left = True
    elif key == glfw.KEY_A and action == glfw.RELEASE:
        left = False
    if key == glfw.KEY_D and action == glfw.PRESS:
        right = True
    elif key == glfw.KEY_D and action == glfw.RELEASE:
        right = False

# do the movement, call this function in the main loop
def do_movement():
    if left:
        cam.process_keyboard("LEFT", 0.05)
    if right:
        cam.process_keyboard("RIGHT", 0.05)
    if forward:
        cam.process_keyboard("FORWARD", 0.05)
    if backward:
        cam.process_keyboard("BACKWARD", 0.05)


# the mouse position callback function
def mouse_look_clb(window, xpos, ypos):
    global first_mouse, lastX, lastY

    if first_mouse:
        lastX = xpos
        lastY = ypos
        first_mouse = False

    xoffset = xpos - lastX
    yoffset = lastY - ypos

    lastX = xpos
    lastY = ypos

    if not mouse_l_pressed:
        cam.process_mouse_movement(xoffset, yoffset)

MAX_DIST = 20
SOFT_SELECTION = True
K_POINTS = 10
if not SOFT_SELECTION:
    K_POINTS = 1

def process_clicks(cloud_coords):
    global last_click_coords, last_click_processed, selected_idx, last_selected_idx
    if not last_click_processed:
        pixel_coords = calculate_pixel_space(cloud_coords)
        last_click_processed = True
        #Filter points that are in the screen:
        pixels_in_screen = (pixel_coords[:,0]<WIDTH) & (pixel_coords[:,0]>0) & (pixel_coords[:,1]<HEIGHT) & (pixel_coords[:,1]>0)
        if pixels_in_screen.sum()>0:
            nn = NearestNeighbors(n_neighbors=K_POINTS, algorithm='ball_tree').fit(pixel_coords[pixels_in_screen])
            dist, idx = nn.kneighbors(np.array(last_click_coords).reshape(1,-1))
            if K_POINTS>1:
                dist = 1/dist
                selected_idx = {np.argwhere(pixels_in_screen)[idxi,0]: d/np.sum(dist) for idxi, d in zip(idx[0], dist[0])}
            else:
                if selected_idx > 0:
                    last_selected_idx = selected_idx
                if dist < MAX_DIST:
                    selected_idx = np.argwhere(pixels_in_screen)[idx][0][0,0]
                else:
                    selected_idx = -1
        if isinstance(selected_idx, dict):
            final_embedding = 0
            for k,v in selected_idx.items():
                final_embedding += v*data['embeddings'][k]
            x,gen = generate_interp(final_embedding, models_data['gpt'],generation_steps=75, temperature=0.6)
            sd.play(x[0].astype(np.float32), 24000)
        
        elif selected_idx > 0:
            filename = data['wav'][selected_idx]
            x,fs = sf.read(filename)
            sd.play(x, fs)

cloud_coords, data = read_pointcloud('test.pkl')
cloud_coords = (cloud_coords - cloud_coords.mean())/cloud_coords.std()
cloud_vertices = flatten_pointcloud_with_color(cloud_coords).astype(np.float32)

def calculate_pixel_space(cloud_coords):
    cloud_coords = np.concatenate([cloud_coords, np.ones((cloud_coords.shape[0],1))], axis=-1)
    pos_pixel = np.matmul(np.matmul(cloud_coords, cam.get_view_matrix()), projection)
    pos_pixel = pos_pixel[:,:2]/pos_pixel[:,3,np.newaxis]
    pos_pixel = (pos_pixel + 1)/2
    pos_pixel[:,0] = pos_pixel[:,0]*WIDTH
    pos_pixel[:,1] = pos_pixel[:,1]*HEIGHT

    return pos_pixel

def mouse_btn_clb(window, button, action, mods):
    global mouse_l_pressed, last_click_processed, last_click_coords
    if (button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS):
        mouse_l_pressed = True
        last_click_coords = (lastX, (HEIGHT - lastY - HEIGHT/5))
        last_click_processed = False

    elif (button == glfw.MOUSE_BUTTON_LEFT and action == glfw.RELEASE):
        mouse_l_pressed = False

cam = Camera()

window = setup_glfw(window_size=(WIDTH,HEIGHT))
glfw.set_key_callback(window, key_input_clb)
glfw.set_cursor_pos_callback(window, mouse_look_clb)
glfw.set_mouse_button_callback(window, mouse_btn_clb)
shader = compile_shader('vertex.gl', 'fragment.gl')

VBO = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, VBO)
glBufferData(GL_ARRAY_BUFFER, cloud_vertices.nbytes, cloud_vertices, GL_STATIC_DRAW)

glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))

glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))

glUseProgram(shader)
glClearColor(0, 0.1, 0.1, 1)
glEnable(GL_DEPTH_TEST)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
glPointSize(10)

projection = pyrr.matrix44.create_perspective_projection_matrix(45, WIDTH/HEIGHT, 0.1, 100)
model = pyrr.matrix44.create_identity()
# eye, target, up


model_loc = glGetUniformLocation(shader, "model")
proj_loc = glGetUniformLocation(shader, "projection")
view_loc = glGetUniformLocation(shader, "view")

glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)

cursor = glfw.create_standard_cursor(glfw.CROSSHAIR_CURSOR)
glfw.set_cursor(window, cursor)

while not glfw.window_should_close(window):
    glfw.poll_events()
    do_movement()
    process_clicks(cloud_coords)
    
    if not isinstance(selected_idx, dict) and (selected_idx > -1):
        glBufferSubData(GL_ARRAY_BUFFER, (selected_idx*6 +3)*4,4*3,np.array([0,1,0], dtype=np.float32))
        if (last_selected_idx > 0) and (last_selected_idx != selected_idx):
            glBufferSubData(GL_ARRAY_BUFFER, (last_selected_idx*6 +3)*4,4*3,np.array([1,0,0], dtype=np.float32))
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    view = cam.get_view_matrix()
    
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
    glDrawArrays(GL_POINTS, 0, int(len(cloud_vertices)/6))

    glfw.swap_buffers(window)

glfw.destroy_cursor(cursor)
glfw.terminate()

