%%% This file is part of the dictionaries-common package.
%%% It has been automatically generated.
%%% DO NOT EDIT!

#ifexists aspell_add_dictionary
  if (_slang_utf8_ok) {
    aspell_add_dictionary (
      "american",
      "en_US",
      "",
      "'",
      "");
    aspell_add_dictionary (
      "british",
      "en_GB",
      "",
      "'",
      "");
    aspell_add_dictionary (
      "canadian",
      "en_CA",
      "",
      "'",
      "");
    aspell_add_dictionary (
      "english",
      "en",
      "",
      "'",
      "");
  } else {
  aspell_add_dictionary (
    "american",
    "en_US",
    "",
    "'",
    "");
  aspell_add_dictionary (
    "british",
    "en_GB",
    "",
    "'",
    "");
  aspell_add_dictionary (
    "canadian",
    "en_CA",
    "",
    "'",
    "");
  aspell_add_dictionary (
    "english",
    "en",
    "",
    "'",
    "");
  }
#endif

#ifexists ispell_add_dictionary
  ispell_add_dictionary (
    "american",
    "american",
    "",
    "'",
    "",
    "-B -d american");
  ispell_add_dictionary (
    "british",
    "british",
    "",
    "'",
    "",
    "-B -d british");
  ispell_add_dictionary (
    "castellano",
    "espa~nol",
    "áéíñóúüÁÉÍÑÓÚÜ",
    "'",
    "~tex",
    "-B");
  ispell_add_dictionary (
    "castellano8",
    "espa~nol",
    "ñáàéèíìóòúùüÑÁÀÉÈÍÌÓÒÚÙÜ",
    "'",
    "~latin1",
    "-B");
  ispell_add_dictionary (
    "francais",
    "french",
    "ÀÂÇÈÉÊËÎÏÔÙÛÜàâçèéêëîïôùûü",
    "-'",
    "~list",
    "");
  ispell_add_dictionary (
    "german-new",
    "ngerman",
    "\"",
    "'",
    "~tex",
    "-C -d ngerman");
  ispell_add_dictionary (
    "german-new8",
    "ngerman",
    "ÄÖÜäößü",
    "'",
    "~latin1",
    "-C -d ngerman");
  ispell_add_dictionary (
    "german-old",
    "ogerman",
    "\"",
    "'",
    "~tex",
    "-C -d ogerman");
  ispell_add_dictionary (
    "german-old8",
    "ogerman",
    "ÄÖÜäößü",
    "'",
    "~latin1",
    "-C -d ogerman");
  ispell_add_dictionary (
    "italiano",
    "italian",
    "ÀÁÈÉÌÍÒÓÙÚàáèéìíóùú",
    "-",
    "~tex",
    "-B -d italian");
  ispell_add_dictionary (
    "swiss",
    "swiss",
    "\"",
    "'",
    "~tex",
    "-C -d swiss");
  ispell_add_dictionary (
    "swiss8",
    "swiss",
    "ÄÖÜäößü",
    "'",
    "~latin1",
    "-C -d swiss");
#endif
