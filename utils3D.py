from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import joblib
import numpy as np
import glfw
import pyrr

def compile_shader(vertex_src, fragment_src):
    with open(vertex_src, 'r') as f:
        vertex_src = f.read()
    with open(fragment_src, 'r') as f:
        fragment_src = f.read()        
    shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))
    return shader

def read_pointcloud(file):
    data = joblib.load(file)
    point_coords = np.stack([data['x'], data['y'], data['z']]).T

    return point_coords, data

def flatten_pointcloud_with_color(x, color=(1,0,0)):
    if len(x) != len(color):
        color = [list(color),]*len(x)
    color = np.array(color)
    point_coords = np.concatenate([x, color], axis=1)
    return point_coords.reshape(-1)

#def window_resize(window, width, height):
#    glViewport(0, 0, width, height)
#    projection = pyrr.matrix44.create_perspective_projection_matrix(45, width / height, 0.1, 100)
    #glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)

def setup_glfw(window_size=(1280,720), window_title='SoundMap'):
    if not glfw.init():
        raise Exception("glfw can not be initialized!")

    window = glfw.create_window(window_size[0], window_size[1], window_title, None, None)

    if not window:
        glfw.terminate()
        raise Exception("glfw window can not be created!")

    glfw.set_window_pos(window, 400, 200) #Hacer que sea centrado
    #glfw.set_window_size_callback(window, window_resize_callback)
    glfw.make_context_current(window)
    return window