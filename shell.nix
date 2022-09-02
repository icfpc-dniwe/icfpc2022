{ nixpkgs ? import <nixpkgs> {}, compiler ? "default" }:

let

  inherit (nixpkgs) pkgs lib;

  haskellPackages_ = if compiler == "default"
                        then pkgs.haskellPackages
                        else pkgs.haskell.packages.${compiler};

  hlib = pkgs.haskell.lib;

  haskellPackages = haskellPackages_.override {
    overrides = self: super: {
    };
  };

  drv = haskellPackages.callPackage ./default.nix {};

  shell = drv.env.overrideAttrs (self: {
    nativeBuildInputs = self.nativeBuildInputs ++ [
      haskellPackages.cabal-install
      haskellPackages.hpack
    ];
  });

  drv_ = drv.overrideAttrs (self: {
    passthru = self.passthru // { shell = shell; };
  });

in

  if lib.inNixShell then drv_.shell else drv_
