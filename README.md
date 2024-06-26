# Ouroboros

## IMP:- Make sure your system is updated

## Install meson and ninja if not installed already

```sh
sudo pacman -S meson
```

## Install the dependencies

```sh
sudo pacman -S gsl openmp
```

## Clone the library

```sh
git clone https://github.com/SaptakBhoumik/Ouroboros.git
cd Ouroboros
```

## Set up build env

### Debug mode

```sh
meson builddir
cd builddir
```

### Release mode 

```sh
meson --buildtype=release dist
cd dist
```

## Build it
Modify the ``test_tensor.cpp`` file and then run
```sh
ninja
```

## Run it

```sh
./test_tensor.elf
```