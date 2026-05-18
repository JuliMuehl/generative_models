import moderngl
import numpy as np
from .utils import *
import os


class GroundTruthRenderer:
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
    layout(location = 0) out vec3 frag_color;
    layout(location = 1) out vec3 frag_direction;
    uniform mat3 u_frame;

    float intersect_ray_sphere(vec3 o, vec3 d, vec3 x, float r){
        o = o - x;
        float a = 1;
        float b = 2*dot(o,d);
        float c = dot(o,o) - r*r;
        float radicant = b*b - 4*a*c;
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

    float intersect_ray_yplane(vec3 o, vec3 d){
        return -o.y/d.y;
    }

    void main(){
        vec3 ray_origin = -u_frame * vec3(0.0, 0.0, 1.0);
        vec2 uv = vec2(frag_uv.x, 1.0 - frag_uv.y);
        vec3 ray_direction = normalize(u_frame * vec3(2.0 * uv - 1.0, 1.0));
        float sphere_radius = 0.3;
        vec3 sphere_origin = vec3(0.0, sphere_radius, 0.0);
        float tsphere = intersect_ray_sphere(ray_origin, ray_direction, sphere_origin, sphere_radius);
        float tplane = intersect_ray_yplane(ray_origin, ray_direction);
        float t = -1.0;
        vec3 col = vec3(1.0);
        float mask = 0.0;
        if(tsphere > 0.0 && (tsphere < t || t < 0)){
            mask = 1.0;
            t = tsphere;
            col = (vec3(1.0) + (ray_origin + t * ray_direction) / sphere_radius) / 2.0;
        }
        if(tplane > 0.0 && (tplane < t || t < 0)){
            t = tplane;
            vec3 xyz = (ray_origin + t * ray_direction);
            if(abs(xyz.x) <= 1.0 && abs(xyz.z) <= 1.0){
                vec2 xz = round(10 * xyz.xz);
                mask = 1.0;
                if(mod(xz.x + xz.y, 2.0) == 1.0){
                    col = vec3(0.8);
                }else{
                    col = vec3(0.1);
                }
            }
        }
        frag_color = col;
        frag_direction = 0.5 * (ray_direction + vec3(1.0));
    }
    """

    def __init__(self, ctx=None):
        if ctx is None:
            self.ctx = moderngl.create_context(standalone=True)
        else:
            self.ctx = ctx
        self.ctx.pixel_pack_alignment = 1
        uv_data = np.array([[0, 0], [1,0], [1,1], [0, 0], [0,1], [1,1]]).astype(np.float32)
        self.uv_buffer = self.ctx.buffer(uv_data)
        self.program = self.ctx.program(self.vertex_shader_source, self.fragment_shader_source)
        self.vao = self.ctx.vertex_array(self.program, self.uv_buffer, "in_uv")
        self.viewport = (512, 512)
        self.color_texture = self.ctx.texture(self.viewport, components=3)
        self.direction_texture = self.ctx.texture(self.viewport, components=3)
        self.color_texture.use(location=0)
        self.direction_texture.use(location=1)
        self.framebuffer = self.ctx.framebuffer(color_attachments=(self.color_texture,self.direction_texture))

    def set_viewport(self, x, y, w, h):
        self.ctx.viewport = (x, y, w, h)

    def render_to_texture(self, x = [0, 0.4, -1], up = [0, 1, 0]):
        self.framebuffer.use()
        frame = tangent_frame(x, up)
        self.program["u_frame"].write(np.ascontiguousarray(frame.T))
        self.vao.render()
        color_shape = (self.viewport[1], self.viewport[0], 3)
        direction_shape = (self.viewport[1], self.viewport[0], 3)
        colors = np.frombuffer(self.color_texture.read(), dtype=np.uint8).reshape(color_shape)
        directions = np.frombuffer(self.direction_texture.read(), dtype=np.uint8).reshape(direction_shape)
        return colors, directions

    def render_to_screen(self, x = [0, 0.4, -1], up = [0, 1, 0]):
        self.ctx.screen.use()
        frame = tangent_frame(x, up)
        self.program["u_frame"].write(np.ascontiguousarray(frame.T))
        self.vao.render()


class VoxelRenderer:
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
    out vec3 frag_color;
    uniform sampler3D u_color_grid;
    uniform sampler3D u_density_grid;
    uniform mat3 u_frame;
    const int num_samples = 32;
    void main(){
        vec3 ray_origin = -u_frame * vec3(0.0, 0.0, 1.0);
        vec2 uv = vec2(frag_uv.x, 1.0 - frag_uv.y);
        vec3 ray_direction = normalize(u_frame * vec3(2.0 * uv - 1.0, 1.0));
        vec3 radiance = vec3(0.0);
        float delta = 2.0 / float(num_samples - 1);
        float transmittance = 1.0;
        for(int i = 0;i<num_samples;i++){
            float t = delta * float(i);
            vec3 x = ray_origin + t * ray_direction;
            vec3 xtex = 0.5 * (x + vec3(1.0));
            vec3 cond = xtex * (1.0 - xtex);
            if(cond.x >= 0 && cond.y >= 0.0 && cond.z >= 0.0){
                float density = texture3D(u_density_grid, xtex).x;
                float alpha = 1.0 - exp(-density*delta);
                vec3 color = texture3D(u_color_grid, xtex).xyz;
                radiance += alpha * transmittance * color;
                transmittance *= (1.0 - alpha);
            }else{
                //Since the box is convex we can stop once we leave it
                break;
            }
        }
        vec3 bg_color = vec3(1.0);
        radiance += transmittance * bg_color;
        frag_color = radiance;
    }
    """

    def __init__(self, ctx=None, grid_file="voxel_data.npz"):
        if ctx is None:
            self.ctx = moderngl.create_context(standalone=True)
        else:
            self.ctx = ctx
        self.ctx.pixel_pack_alignment = 1
        uv_data = np.array([[0, 0], [1,0], [1,1], [0, 0], [0,1], [1,1]]).astype(np.float32)
        self.uv_buffer = self.ctx.buffer(uv_data)
        self.program = self.ctx.program(self.vertex_shader_source, self.fragment_shader_source)
        self.vao = self.ctx.vertex_array(self.program, self.uv_buffer, "in_uv")
        self.color_grid_texture = None
        self.density_grid_texture = None
        self.grid_file = grid_file
        self.grid_mtime = None
        self.load_textures()

    def set_viewport(self, x, y, w, h):
        self.ctx.viewport = (x, y, w, h)

    def load_textures(self):
        if self.grid_mtime is None or os.path.getmtime(self.grid_file) > self.grid_mtime:
            with np.load(self.grid_file) as voxel_data:
                color_grid, density_grid = voxel_data["color_grid"], voxel_data["density_grid"]
                if self.color_grid_texture is None or self.density_grid_texture is None:
                    self.color_grid_texture = self.ctx.texture3d(size=color_grid.shape[:-1], components = color_grid.shape[-1], dtype="f4")
                    self.density_grid_texture = self.ctx.texture3d(size=density_grid.shape[:-1], components = 1, dtype="f4")
                    self.color_grid_texture.filter = (moderngl.LINEAR,moderngl.LINEAR)
                    self.density_grid_texture.filter = (moderngl.LINEAR,moderngl.LINEAR)
                self.color_grid_texture.write(color_grid.astype(np.float32))
                self.density_grid_texture.write(density_grid.astype(np.float32))
            self.grid_mtime = os.path.getmtime(self.grid_file)

    def render_to_screen(self, x = [0, 0.4, -1], up = [0, 1, 0]):
        self.ctx.screen.use()
        frame = tangent_frame(x, up)
        self.program["u_frame"].write(np.ascontiguousarray(frame.T))
        self.program["u_color_grid"] = 0
        self.color_grid_texture.use(0)
        self.program["u_density_grid"] = 1
        self.density_grid_texture.use(1)
        self.vao.render()
