{ mkDerivation, aeson, base, bytestring, containers, JuicyPixels
, lib, linear, massiv, mtl, opencv, text, vector
}:
mkDerivation {
  pname = "icfpc2022";
  version = "0.1.0.0";
  src = ./.;
  isLibrary = true;
  isExecutable = true;
  libraryHaskellDepends = [
    aeson base bytestring containers JuicyPixels linear massiv mtl
    opencv text vector
  ];
  executableHaskellDepends = [
    aeson base bytestring containers JuicyPixels linear massiv mtl
    opencv text vector
  ];
  license = "unknown";
  mainProgram = "solve";
}
