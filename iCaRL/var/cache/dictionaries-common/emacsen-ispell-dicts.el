;;; This file is part of the dictionaries-common package.
;;; It has been automatically generated.
;;; DO NOT EDIT!

;; Adding aspell dicts

(add-to-list 'debian-aspell-only-dictionary-alist
  '("english"
     "[a-zA-Z]"
     "[^a-zA-Z]"
     "[']"
     nil
     ("-d" "en")
     nil
     iso-8859-1))
(add-to-list 'debian-aspell-only-dictionary-alist
  '("canadian"
     "[a-zA-Z]"
     "[^a-zA-Z]"
     "[']"
     nil
     ("-d" "en_CA")
     nil
     iso-8859-1))
(add-to-list 'debian-aspell-only-dictionary-alist
  '("british"
     "[a-zA-Z]"
     "[^a-zA-Z]"
     "[']"
     nil
     ("-d" "en_GB")
     nil
     iso-8859-1))
(add-to-list 'debian-aspell-only-dictionary-alist
  '("american"
     "[a-zA-Z]"
     "[^a-zA-Z]"
     "[']"
     nil
     ("-d" "en_US")
     nil
     iso-8859-1))


;; Adding hunspell dicts

(add-to-list 'debian-hunspell-only-dictionary-alist
  '("swiss8"
     "[A-Za-z\304\334\326\344\374\366\337]"
     "[^A-Za-z\304\334\326\344\374\366\337]"
     "[-]"
     nil
     ("-d" "de_CH")
     nil
     iso-8859-1))
(add-to-list 'debian-hunspell-only-dictionary-alist
  '("german-new8"
     "[A-Za-z\304\334\326\344\374\366\337]"
     "[^A-Za-z\304\334\326\344\374\366\337]"
     "[-]"
     nil
     ("-d" "de_DE")
     nil
     iso-8859-1))
(add-to-list 'debian-hunspell-only-dictionary-alist
  '("english_american"
     "[a-zA-Z]"
     "[^a-zA-Z]"
     "[']"
     nil
     ("-d" "en_US")
     nil
     utf-8))
(add-to-list 'debian-hunspell-only-dictionary-alist
  '("de_DE"
     "[A-Za-z\304\334\326\344\374\366\337]"
     "[^A-Za-z\304\334\326\344\374\366\337]"
     "[-]"
     nil
     ("-d" "de_DE")
     nil
     iso-8859-1))
(add-to-list 'debian-hunspell-only-dictionary-alist
  '("de_CH"
     "[A-Za-z\304\334\326\344\374\366\337]"
     "[^A-Za-z\304\334\326\344\374\366\337]"
     "[-]"
     nil
     ("-d" "de_CH")
     nil
     iso-8859-1))


;; Adding ispell dicts

(add-to-list 'debian-ispell-only-dictionary-alist
  '("swiss8"
     "[A-Za-zÄÖÜäößü]"
     "[^A-Za-zÄÖÜäößü]"
     "[']"
     nil
     ("-C" "-d" "swiss")
     "~latin1"
     iso-8859-1))
(add-to-list 'debian-ispell-only-dictionary-alist
  '("swiss"
     "[A-Za-z\"]"
     "[^A-Za-z\"]"
     "[']"
     nil
     ("-C" "-d" "swiss")
     "~tex"
     iso-8859-1))
(add-to-list 'debian-ispell-only-dictionary-alist
  '("italiano"
     "[A-Z\300\301\310\311\314\315\322\323\331\332a-z\340\341\350\351\354\355\363\371\372]"
     "[^A-Z\300\301\310\311\314\315\322\323\331\332a-z\340\341\350\351\354\355\363\371\372]"
     "[-]"
     nil
     ("-B" "-d" "italian")
     "~tex"
     iso-8859-1))
(add-to-list 'debian-ispell-only-dictionary-alist
  '("german-old8"
     "[A-Za-zÄÖÜäößü]"
     "[^A-Za-zÄÖÜäößü]"
     "[']"
     nil
     ("-C" "-d" "ogerman")
     "~latin1"
     iso-8859-1))
(add-to-list 'debian-ispell-only-dictionary-alist
  '("german-old"
     "[A-Za-z\"]"
     "[^A-Za-z\"]"
     "[']"
     nil
     ("-C" "-d" "ogerman")
     "~tex"
     iso-8859-1))
(add-to-list 'debian-ispell-only-dictionary-alist
  '("german-new8"
     "[A-Za-zÄÖÜäößü]"
     "[^A-Za-zÄÖÜäößü]"
     "[']"
     nil
     ("-C" "-d" "ngerman")
     "~latin1"
     iso-8859-1))
(add-to-list 'debian-ispell-only-dictionary-alist
  '("german-new"
     "[A-Za-z\"]"
     "[^A-Za-z\"]"
     "[']"
     nil
     ("-C" "-d" "ngerman")
     "~tex"
     iso-8859-1))
(add-to-list 'debian-ispell-only-dictionary-alist
  '("francais"
     "[A-Za-zÀÂÇÈÉÊËÎÏÔÙÛÜàâçèéêëîïôùûü]"
     "[^A-Za-zÀÂÇÈÉÊËÎÏÔÙÛÜàâçèéêëîïôùûü]"
     "[-']"
     t
     ("-d" "french")
     "~list"
     iso-8859-1))
(add-to-list 'debian-ispell-only-dictionary-alist
  '("castellano8"
     "[a-z\340\341\350\351\354\355\362\363\371\372\374\347\361A-Z\300\301\310\311\314\315\322\323\331\332\334\307\321]"
     "[^a-z\340\341\350\351\354\355\362\363\371\372\374\347\361A-Z\300\301\310\311\314\315\322\323\331\332\334\307\321]"
     "[']"
     nil
     ("-B" "-d" "espa~nol")
     "~latin1"
     iso-8859-1))
(add-to-list 'debian-ispell-only-dictionary-alist
  '("castellano"
     "[a-z\301\311\315\321\323\332\334A-Z\341\351\355\361\363\372\374]"
     "[^a-z\301\311\315\321\323\332\334A-Z\341\351\355\361\363\372\374]"
     "[']"
     nil
     ("-B" "-d" "espa~nol")
     "~tex"
     iso-8859-1))
(add-to-list 'debian-ispell-only-dictionary-alist
  '("british"
     "[A-Za-z]"
     "[^A-Za-z]"
     "[']"
     nil
     ("-B" "-d" "british")
     nil
     iso-8859-1))
(add-to-list 'debian-ispell-only-dictionary-alist
  '("american"
     "[A-Za-z]"
     "[^A-Za-z]"
     "[']"
     nil
     ("-B" "-d" "american")
     nil
     iso-8859-1))



;; No emacsen-aspell-equivs entries were found


;; An alist that will try to map hunspell locales to emacsen names

(setq debian-hunspell-equivs-alist '(
     ("de_CH" "swiss8")
     ("de_DE" "german-new8")
))

;; Get default value for debian-hunspell-dictionary. Will be used if
;; spellchecker is hunspell and ispell-local-dictionary is not set.
;; We need to get it here, after debian-hunspell-equivs-alist is loaded

(setq debian-hunspell-dictionary (debian-ispell-get-hunspell-default))

