#include <iostream>
#include "../libs/bitmap/bitmap_image.hpp"

struct Args
{
    char *input;
    char *output;
    char *mode;
};

Args parseArgs(int argc, char *argv[]);

int main(int argc, char *argv[])
{
    std::cout << "Running CPU ImageRgbToGray program..." << std::endl;
    Args args = parseArgs(argc, argv);

    std::cout << "Loading image from '" << args.input << "'..." << std::endl;
    bitmap_image img(args.input);
    if (!img)
    {
        std::cerr << "Failed to open file '" << args.input << "'." << std::endl;
        exit(1);
    }
    const int h = img.height();
    const int w = img.width();

    std::cout << "Converting image with mode '" << args.mode << "'..." << std::endl;
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            rgb_t pixel = img.get_pixel(i, j);
            int value;
            if (strcmp(args.mode, "lightness") == 0)
            {
                int min = __min(pixel.red, __min(pixel.green, pixel.blue));
                int max = __max(pixel.red, __max(pixel.green, pixel.blue));
                value = min + (max - min) / 2;
            }
            else if (strcmp(args.mode, "average") == 0)
            {
                value = (pixel.red + pixel.green + pixel.blue) / 3;
            }
            else if (strcmp(args.mode, "luminosity") == 0)
            {
                value = (0.3 * pixel.red + 0.59 * pixel.green + 0.11 * pixel.blue);
            }
            else
            {
                std::cerr << "Unsupported mode '" << args.mode << "'." << std::endl;
                exit(1);
            }

            img.set_pixel(i, j, value, value, value);
        }
    }

    std::cout << "Outputting image to '" << args.output << "'..." << std::endl;
    img.save_image(args.output);
}

Args parseArgs(int argc, char *argv[])
{
    // Expect 3 or 4 total args
    if (argc < 3 || argc > 4)
    {
        std::cerr << "Must provide an input filename, output filename, and "
                  << "optionally a grayscale mode to use (lightness, average, and luminosity (default))." << std::endl;
        exit(1);
    }

    Args args = {argv[1], argv[2]};
    if (argc == 4)
        args.mode = argv[3];
    else
        args.mode = (char *)"luminosity";
    return args;
}
