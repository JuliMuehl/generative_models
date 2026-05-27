#version 420 
// ============================================================================
// Original Shader Title: [Ray Marching Experiment 43]
// Author: [Stephane Cuillerdier / Shadertoy aiekick]
// Source: [Shadertoy URL, https://www.shadertoy.com/view/lstXRl]
// 
// Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported
// (CC BY-NC-SA 3.0) https://creativecommons.org/licenses/by-nc-sa/3.0/
// ============================================================================

in vec2 frag_uv;
layout(location = 0) out vec3 frag_color;
layout(location = 1) out vec3 frag_direction;
uniform mat3 u_frame;

float dstepf = 0.0;

vec4 displ(vec3 p)
{
    vec2 g = p.xy;
    vec3 col = vec3(sin(4.0 * g.y * 2.0 * 3.141569), cos(8.0 * g.x * 2.0 * 3.141569), 0.0);
   	col = clamp(col, 0., 1.);
    float dist = dot(col,vec3(0.1));
    return vec4(dist,col);
}

vec4 map(vec3 p)
{
    vec4 disp1 = displ(p*.1);
    vec4 disp2 = displ(p*.2);
    float m = length(p);
    float me = m - 4.78 + disp1.x;
    float mi = m - 4.5 - disp1.x;
    float mk = m - 4.5 + disp2.x;
    float mei = max(-mi, me);
    float dp = 1000000000000.0;
    if(abs(p.x) < 20.0 && abs(p.z) < 20.0){
        dp = p.y;
    }
    if(dp < mk && dp < mei){
        dstepf += 0.015;
        vec2 ij = round(0.5 * p.xz);
        float xor = mod(ij.x, 2.0) *  (1.0 - mod(ij.y, 2.0)) + mod(ij.y, 2.0) *  (1.0 - mod(ij.x, 2.0));
        vec3 col = vec3(xor);
        return vec4(dp, col);
    }
    if (mk < mei) 
    {
        dstepf += 0.025;
        return vec4(mk, disp2.y*vec3(0.2,0.5,0.2));
    }
    dstepf += 0.015;
    return vec4(mei, disp1.y*vec3(0.5,0.2,0.5));
}

///////////////////////////////////////////
//FROM IQ Shader https://www.shadertoy.com/view/Xds3zN
float softshadow( in vec3 ro, in vec3 rd, in float mint, in float tmax )
{
	float res = 1.0;
    float t = mint;
    for( int i=0; i<16; i++ )
    {
		float h = map( ro + rd*t ).x;
        res = min( res, 8.0*h/t );
        t += clamp( h, 0.02, 0.10 );
        if( h<0.001 || t>tmax ) break;
    }
    return clamp( res, 0.0, 1.0 );
}

vec3 calcNormal( in vec3 pos )
{
    if(abs(pos.y) > 0.001){
	vec3 eps = vec3( 0.03, 0., 0. );
	vec3 nor = vec3(
	    map(pos+eps.xyy).x - map(pos-eps.xyy).x,
	    map(pos+eps.yxy).x - map(pos-eps.yxy).x,
	    map(pos+eps.yyx).x - map(pos-eps.yyx).x );
	return normalize(nor);
    }else{
        return vec3(0.0, 1.0, 0.0);
    }
}

float calcAO( in vec3 pos, in vec3 nor )
{
	float occ = 0.0;
    float sca = 1.0;
    for( int i=0; i<5; i++ )
    {
        float hr = 0.01 + 0.12*float(i)/4.0;
        vec3 aopos =  nor * hr + pos;
        float dd = map( aopos ).x;
        occ += -(dd-hr)*sca;
        sca *= 0.95;
    }
    return clamp( 1.0 - 3.0*occ, 0.0, 1.0 );    
}

///////////////////////////////////////////
float march(vec3 ro, vec3 rd, float rmPrec, float maxd, float mapPrec)
{
    float s = rmPrec,so=s;
    float d = 0.;
    for(int i=0;i<250;i++)
    {      
        if (s<rmPrec||s>maxd) break;
        vec3 p = ro+rd*d;
        s = map(p).x;
        s *= (s>so?1.5:1.);so=s; // Enhanced Sphere Tracing => lgdv.cs.fau.de/get/2234 
        d += s * mapPrec;
    }
    return d;
}

vec3 sun(vec3 n){
    const vec3 l = normalize(vec3(1.0, 1.0, 1.0));
    const vec3 ambient = vec3(0.3);
    return ambient + pow(clamp(0.0, dot(l,normalize(n)), 1.0), 4.0) * vec3(1.0);
}

////////MAIN///////////////////////////////
void main()
{
  	float li = 0.6; // light intensity
    float prec = 0.00001; // ray marching precision
    float maxd = 50.; // ray marching distance max
    float refl_i = 0.45; // reflexion intensity
    float refr_a = 0.7; // refraction angle
    float refr_i = 0.8; // refraction intensity
    float bii = 0.35; // bright init intensity
    float marchPrecision = 0.5; // ray marching tolerance precision
    float cam_d = 10.0; // mouse y axis 
    /////////////////////////////////////////////////////////
    
    vec3 col = vec3(0.);

    vec3 ray_origin = -16.0 * u_frame * vec3(0.0, 0.0, 1.0);
    vec2 uv = vec2(frag_uv.x, 1.0 - frag_uv.y);
    vec3 ray_direction = normalize(u_frame * vec3(2.0 * uv - 1.0, 1.0));
    
    vec3 ro = ray_origin; 
  	vec3 rd = ray_direction;
    
    float b = bii;
    
    float d = march(ro, rd, prec, maxd, marchPrecision);
    
    if (d<maxd)
    {
        vec2 e = vec2(-1., 1.)*0.005; 
    	vec3 p = ro+rd*d;
        vec3 n = calcNormal(p);
        
        b=li;
        
        vec3 reflRay = reflect(rd, n);
		vec3 refrRay = refract(rd, n, refr_a);
        
        //vec3 cubeRefl = sun(reflRay) * refl_i;
        //vec3 cubeRefr = sun(refrRay) * refr_i;
        //col = cubeRefl + cubeRefr;
        col = vec3(1.0);
        
        
        
       	// lighting        
        float occ = calcAO( p, n );
		vec3  lig = normalize( vec3(-0.6, 0.7, -0.5) );
		float amb = clamp( 0.5+0.5*n.y, 0.0, 1.0 );
        float dif = clamp( dot( n, lig ), 0.0, 1.0 );
        float bac = clamp( dot( n, normalize(vec3(-lig.x,0.0,-lig.z))), 0.0, 1.0 )*clamp( 1.0-p.y,0.0,1.0);
        float dom = smoothstep( -0.1, 0.1, reflRay.y );
        float fre = pow( clamp(1.0+dot(n,rd),0.0,1.0), 2.0 );
		float spe = pow(clamp( dot( reflRay, lig ), 0.0, 1.0 ),16.0);
        
        
        dif *= softshadow( p, lig, 0.02, 2.5 );
       	dom *= softshadow( p, reflRay, 0.02, 2.5 );

        vec3 brdf = vec3(0.0);
        brdf += 1.20*dif*vec3(1.00,0.90,0.60);
        brdf += 1.20*spe*vec3(1.00,0.90,0.60)*dif;
        brdf += 0.30*amb*vec3(0.50,0.70,1.00)*occ;
        brdf += 0.40*dom*vec3(0.50,0.70,1.00)*occ;
        brdf += 0.30*bac*vec3(0.25,0.25,0.25)*occ;
        brdf += 0.40*fre*vec3(1.00,1.00,1.00)*occ;
        brdf += 0.02;
        col = col*brdf;
      

    	col = mix( col, vec3(0.8,0.9,1.0), 1.0-exp( -0.0005*d*d ) );
        
       	col = mix(col, map(p).yzw, 0.5);
    }else{
        col = vec3(1.0);
    }
	frag_color.rgb = col;
    frag_direction = 0.5 * (ray_direction + vec3(1.0));
}