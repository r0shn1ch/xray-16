vec3 bloom_highlights(vec3 color, float threshold, float knee)
{
    float brightness = max(dot(color, vec3(0.2126, 0.7152, 0.0722)), float(0.0));
    float safe_knee = max(knee, float(0.0001));
    float soft = clamp(brightness - threshold + safe_knee, float(0.0), float(2.0) * safe_knee);
    soft = soft * soft / (float(4.0) * safe_knee);
    float contribution = max(brightness - threshold, float(0.0)) + soft;
    return color * (contribution / max(brightness, float(0.0001)));
}
