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
