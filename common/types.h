struct alignas(float) Detection {
  float bbox[4];  // xmin ymin xmax ymax
  float conf;  // 
  float class_id;
};
struct Point {
    int x;
    int y;
};