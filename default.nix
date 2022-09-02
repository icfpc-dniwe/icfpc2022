{ mkDerivation, aeson, base, bytestring, JuicyPixels, lib, linear
, text, vector
}:
mkDerivation {
  pname = "icfpc2022";
  version = "0.1.0.0";
  src = ./.;
  isLibrary = true;
  isExecutable = true;
  libraryHaskellDepends = [
    aeson base bytestring linear text vector
  ];
  executableHaskellDepends = [
    aeson base bytestring JuicyPixels linear text vector
  ];
  license = "unknown";
  mainProgram = "solve";
}
