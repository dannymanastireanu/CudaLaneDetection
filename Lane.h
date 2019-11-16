#pragma once



class Lane
{

private:

	int width;
	int hight;
	unsigned char* image;
	bool draw_on_left;
	bool draw_on_right;

public:
	Lane();
	Lane(int width, int hight, unsigned char* data_image);
	~Lane();
};

