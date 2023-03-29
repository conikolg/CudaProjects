#include <iostream>
#include "../libs/bitmap/bitmap_image.hpp"

struct Args
{
    char* input;
    char* output;
    char* mode;
};

Args parseArgs(int argc, char *argv[]);


int main(int argc, char *argv[])
{
    std::cout << "Running CPU ImageRgbToGray program..." << std::endl;
    Args args = parseArgs(argc, argv);

    std::cout << "Loading image..." << std::endl;
    bitmap_image img(args.input);
    if (!img)
    {
        std::cerr << "Failed to open file '" << args.input << "'." << std::endl;
        exit(1);
    }
    const int h = img.height();
    const int w = img.width();
    rgb_t pixels[h][w];
    for(int i = 0; i < h; i++)
        for(int j = 0; j < w; j++)
            img.get_pixel(i, j, pixels[i][j]);

    std::cout << "Converting image..." << std::endl;

}

Args parseArgs(int argc, char *argv[]) {
    // Expect 3 or 4 total args
    if (argc < 3 || argc > 4)
    {
        std::cerr << "Must provide an input filename, output filename, and " 
            << "optionally a grayscale mode to use (one of lightness, average, and luminosity)." << std::endl;
        exit(1);
    }

    return {
        argv[1], 
        argv[2], 
        (argc == 4) ? argv[3] : nullptr
    };
}
