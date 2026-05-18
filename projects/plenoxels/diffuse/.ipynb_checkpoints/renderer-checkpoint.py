import moderngl
import numpy as np
from PIL import Image

def tangent_frame(theta, phi, dtype=np.float32):
        st, ct, sp, cp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
        return np.array([
            [ ct*cp,-st*sp,-st*cp],
            [ ct*sp, st*cp,-st*sp],
            [-sp   , 0    ,-ct   ]
        ]).astype(dtype=dtype)

class Renderer:
    vertex_shader_source = """
    #version 330
    in vec2 in_uv;
    out vec2 frag_uv;
    void main(){
        frag_uv = in_uv;
        gl_Position = vec4(2.0 * in_uv - 1.0, 0.0, 1.0);
    }
    """
    fragment_shader_source = """
    #version 330
    in vec2 frag_uv;
    out vec4 frag_color;
    uniform mat3 u_frame;

    float intersect_ray_sphere(vec3 o, vec3 d, vec3 x, float r){
        float a = dot(o,o) - r*r;
        float b = 2*dot(o,d);
        float c = 1;
        float radicant = 4*a*c - b*b;
        if(radicant > 0){
            float s = sqrt(radicant);
            float t1 = (-b + s) / (2*a);
            float t2 = (-b - s) / (2*a);
            if(t1 < 0) return t2;
            if(t2 < 0) return t1;
            return min(t1, t2);
        }
        return -1.0;
    }

    void main(){
        vec3 ray_origin = -u_frame * vec3(0.0, 0.0, 1.0);
        vec3 ray_direction = normalize(u_frame * vec3(2.0 * frag_uv - 1.0, 1.0));
        float sphere_radius = 0.2;
        vec3 sphere_origin = vec3(0.0);
        float t = intersect_ray_sphere(ray_origin, ray_direction, sphere_origin, sphere_radius);
        if(t > 0){
            frag_color = vec4(1.0);
        }else{
            frag_color = vec4(vec3(0.0), 0.0);
        }
    }
    """

    def __init__(self):
        self.ctx = moderngl.create_context(standalone=True)
        self.ctx.pixel_pack_alignment = 1
        uv_data = np.array([[0, 0], [1,0], [1,1], [0, 0], [0,1], [1,1]]).astype(np.float32)
        self.uv_buffer = self.ctx.buffer(uv_data)
        self.program = self.ctx.program(self.vertex_shader_source, self.fragment_shader_source)
        self.vao = self.ctx.vertex_array(self.program, self.uv_buffer, "in_uv")
        self.viewport = (512, 512)
        self.color_texture = self.ctx.texture(self.viewport, components=3)
        self.framebuffer = self.ctx.framebuffer(self.color_texture)

    def render(self,theta=np.pi/4,phi=0):
        self.framebuffer.use()
        self.program.get("u_frame", default=None).write(tangent_frame(theta,phi))
        self.vao.render()
        output_shape = (self.viewport[1], self.viewport[0], 3)
        return np.frombuffer(self.color_texture.read(), dtype=np.uint8).reshape(output_shape)
