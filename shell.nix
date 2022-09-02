{ nixpkgs ? import <nixpkgs> {}, compiler ? "default" }:

let

  inherit (nixpkgs) pkgs;

  haskellPackages = pkgs.haskell.packages.ghc941;

  haskellPackages_ = haskellPackages.override {
    overrides = self: super: {
    };
  };

  drv = haskellPackages_.callPackage ./default.nix {};

in

  if pkgs.lib.inNixShell then drv.env else drv
