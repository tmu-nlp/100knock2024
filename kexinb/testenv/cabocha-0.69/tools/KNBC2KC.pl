#!/usr/bin/perl -w

# Convert KNBC corpus to Kyoto Text Corpus-style
# Bunsetsu-level tree.

use strict;
use warnings;

my @chunks;
my @chunk;
my %token2chunk;
my %token2dep;
my $link = -1;
my $token = 0;

while (<>) {
    chomp;
    next if (/^\#/);
    if (/^EOS/) {
	push @chunks, [[@chunk], $link];
	for (my $c = 0; $c <= $#chunks; ++$c) {
	    my ($chunk, $link) = @{$chunks[$c]};
	    my $clink = -1;
	    if ($c == $#chunks) {
		$clink = -1;
	    } else {
		while (1) {
		    if ($link == -1 || !defined $link) {
                        $clink= -1;
                        last;
                    }
		    $clink = $token2chunk{$link};
		    last if ($clink > $c);
		    $link = $token2dep{$link};
		}
	    }
	    print "* $c $clink" , "D\n";
	    for (@{$chunk}) {
		print "$_\n";
	    }
	}
	print "EOS\n";
	%token2chunk = ();
        %token2dep = ();
	@chunks = ();
	@chunk = ();
	$token = 0;
	$link = -1;
    } elsif (/^\* ([-\d]+)(.+)/) {
	my $old_link = $1;
	if (scalar(@chunk) > 0) {
	    die if ($link == -1);
	    push @chunks, [[@chunk], $link];
	}
	$link = $old_link;
	@chunk = ();
    } elsif (/^[\+\*] ([-\d]+)(.+)/) {
    } else {
	# 使って つかって 使う 動詞 2 * 0 子音動詞ワ行 12 タ系連用テ形
	# 民主 みんしゅ * 名詞 普通名詞 * *
	# ［ ［ ［ 特殊 1 括弧始
	my @a = split;
	push @chunk, "$a[0]\t$a[1]";
#	push @chunk, "$a[0] $a[1] * $a[3] $a[5] $a[7] $a[9]";
    }

    if (/^[\+\*] ([-\d]+)(.+)/) {
	my $link = $1;
	my $n = scalar(@chunks);
	$token2chunk{$token} = $n;
	$token2dep{$token} = $link;
	++$token;
    }
}

