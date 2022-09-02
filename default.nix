{ mkDerivation, aeson, base, bytestring, lib, text }:
mkDerivation {
  pname = "icfpc2022";
  version = "0.1.0.0";
  src = ./.;
  isLibrary = true;
  isExecutable = true;
  libraryHaskellDepends = [ aeson base bytestring text ];
  executableHaskellDepends = [ aeson base bytestring text ];
  doHaddock = false;
  license = "unknown";
  mainProgram = "client";
}
