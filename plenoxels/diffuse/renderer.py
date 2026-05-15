import moderngl
import numpy as np
from tqdm import tqdm

def tangent_frame(x, up, dtype=np.float32):
        res = np.eye(3, dtype=dtype)
        res[:, 2] = -np.array(x, dtype=dtype)
        res[:, 0] = np.cross(res[:, 2], up)
        res[:, 1] = np.cross(res[:, 2], res[:, 0])
        return res / np.linalg.norm(res, axis=0)

def unit_sphere(theta, phi, dtype=np.float32):
    return [
        np.sin(theta) * np.cos(phi),
        np.cos(theta),
        np.sin(theta) * np.sin(phi),
    ]

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
        vec3 ray_direction = normalize(u_frame * vec3(2.0 * frag_uv - 1.0, 1.0));
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

    def render(self, x = [0, 0.4, -1], up = [0, 1, 0]):
        self.framebuffer.use()
        frame = tangent_frame(x, up)
        self.program.get("u_frame", default=None).write(np.ascontiguousarray(frame.T))
        self.vao.render()
        color_shape = (self.viewport[1], self.viewport[0], 3)
        direction_shape = (self.viewport[1], self.viewport[0], 3)
        colors = np.frombuffer(self.color_texture.read(), dtype=np.uint8).reshape(color_shape)
        directions = np.frombuffer(self.direction_texture.read(), dtype=np.uint8).reshape(direction_shape)
        return colors, directions

class VoxelGridRenderer:
    def __init__(self, density_grid, color_grid, ctx=None):
        if ctx is None:
            self.ctx = moderngl.create_context(standalone=True)
        else:
            self.ctx = ctx
        self.density_texture = ctx.texutre3d(density_grid.shape[1:], components=density_grid.shape[0], data=density_grid, dtype="f4")
        self.color_grid = ctx.texture3d(color_crid.shape[1:], components=color_grid.shape[0], data=color_grid, dtype="f4")

if __name__ == "__main__":
    import json
    from PIL import Image
    import os

    def render_and_save(renderer, image_dir, positions):
        if not os.path.exists(image_dir):
            os.mkdir(image_dir)
        for i, pos in tqdm(list(enumerate(positions))):
            colors, directions= renderer.render(x = pos)
            colors, directions= Image.fromarray(colors), Image.fromarray(directions)
            colors.save(os.path.join(image_dir, f"img{i}.png"))
            directions.save(os.path.join(image_dir,f"dirs{i}.png"))
        with open(os.path.join(image_dir, "camera_poses.json"), "w") as f:
            json.dump({"camera_positions":positions},f)
    
    renderer = GroundTruthRenderer()
    train_positions = [unit_sphere(theta,phi) for theta in np.linspace(np.pi/8, np.pi/3, 16) for phi in np.linspace(0, 2 * np.pi, 16)]
    train_image_dir = "train_images"
    print("Generating Training Images:")
    render_and_save(renderer, train_image_dir, train_positions)
    test_positions = [unit_sphere(theta+0.1 * np.random.random(),phi + np.random.randn()) for theta in np.linspace(np.pi/8, np.pi/3, 8) for phi in np.linspace(0, 2 * np.pi, 8)]
    test_image_dir = "test_images"
    print("Generating Test  Images:")
    render_and_save(renderer, test_image_dir, test_positions)
